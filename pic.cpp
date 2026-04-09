#include "pic.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace ImgParse
{
	using namespace cv;
	using namespace std;

	namespace
	{
		constexpr int   kFrameSize = 266;

		// ── Layout constants (must match code.cpp / ImgDecode.cpp exactly) ──────────
		// Header: 3 rows × 16 columns of 1-pixel-wide bits.
		constexpr int kHdrTop  = 6;
		constexpr int kHdrLeft = 42;
		constexpr int kHdrH    = 3;
		constexpr int kHdrW    = 16;

		// Data areas (rows, cols, h, w) — identical to code.cpp kDataAreas.
		struct DataArea { int top, left, h, w; };
		constexpr DataArea kDataAreas[5] = {
			{6,   58, 3,   166},   // top-right strip
			{9,   42, 33,  182},   // upper-middle band
			{42,  5,  179, 256},   // main data block
			{221, 5,  3,   256},   // lower-middle band
			{224, 42, 37,  182},   // bottom strip
		};

		// Corner data cell region: rows/cols in [kCornerStart, kFrameSize),
		// minus the quiet zone (row≥kSqrQuiet or col≥kSqrQuiet) and the
		// safety zone ([kSqrSafetyS, kSqrSafetyE] × [kSqrSafetyS, kSqrSafetyE]).
		// All thresholds derived from code.cpp constants (SmallQrPointbias=14, Radius=6,
		// CornerReserveSize=42).
		constexpr int kCornerStart  = kFrameSize - 42; // 224
		constexpr int kSqrQuiet     = 260;             // SmallQrPointEnd + 1 = 259 + 1
		constexpr int kSqrSafetyS   = 242;             // SmallQrPointStart - 4
		constexpr int kSqrSafetyE   = 263;             // SmallQrPointEnd   + 4

		// Collect pixel values from every signal-bearing position in a 266×266 grayscale
		// image for use as the Otsu histogram.  Excludes the safe-area border, the three
		// large finder patterns, and the small BR finder — so the threshold is calibrated
		// purely to the data black/white distribution and not biased by structure pixels.
		double dataOtsuThresh(const Mat& gray266)
		{
			std::vector<uint8_t> vals;
			vals.reserve(64000);

			// Header pixels.
			for (int r = kHdrTop; r < kHdrTop + kHdrH; ++r)
				for (int c = kHdrLeft; c < kHdrLeft + kHdrW; ++c)
					vals.push_back(gray266.at<uint8_t>(r, c));

			// Five data areas.
			for (const auto& a : kDataAreas)
				for (int r = a.top; r < a.top + a.h && r < kFrameSize; ++r)
					for (int c = a.left; c < a.left + a.w && c < kFrameSize; ++c)
						vals.push_back(gray266.at<uint8_t>(r, c));

			// Corner data cells (replicates code.cpp buildCornerDataCells logic).
			for (int r = kCornerStart; r < kFrameSize; ++r)
				for (int c = kCornerStart; c < kFrameSize; ++c)
				{
					if (r >= kSqrQuiet || c >= kSqrQuiet) continue;
					if (r >= kSqrSafetyS && r <= kSqrSafetyE
						&& c >= kSqrSafetyS && c <= kSqrSafetyE) continue;
					vals.push_back(gray266.at<uint8_t>(r, c));
				}

			if (vals.empty()) return 127.0;
			const Mat m(1, static_cast<int>(vals.size()), CV_8U, vals.data());
			Mat tmp;
			return threshold(m, tmp, 0, 255, THRESH_BINARY | THRESH_OTSU);
		}

		// Apply a fixed threshold to every pixel and write the result as a
		// 266×266 CV_8UC3 pure-black-or-white image ready for ImageDecode::Main.
		void applyPixelBinary(const Mat& gray266, double thresh, Mat& out)
		{
			out.create(kFrameSize, kFrameSize, CV_8UC3);
			const uint8_t t = static_cast<uint8_t>(thresh);
			for (int r = 0; r < kFrameSize; ++r)
				for (int c = 0; c < kFrameSize; ++c)
					out.at<Vec3b>(r, c) = (gray266.at<uint8_t>(r, c) > t)
						? Vec3b(255, 255, 255)
						: Vec3b(0, 0, 0);
		}

		struct Marker
		{
			Point2f center;
			float   area;
		};

		// Cached corners from the most recent successful detection.
		struct CachedCorners { Point2f tl, tr, br, bl; bool valid = false; };
		CachedCorners g_cachedCorners;
		int g_lastCols = 0;
		int g_lastRows = 0;
		int g_frameCount = 0;

		// ── Localization: ported from modify/warp_engine.cpp (FindMarkerCenters) ──
		// Runs on the pre-scaled image (≤800 px max dim).
		// Detects finder-pattern markers via 2-level nested contours, HSV saturation
		// suppression, adaptive Gaussian threshold, and morphological close — exactly
		// as implemented in the original working engine. Parameters are intentionally
		// unchanged from warp_engine.cpp.
		vector<Marker> findMarkers(const Mat& img)
		{
			Mat gray;
			if (img.channels() == 3)
			{
				Mat hsv, satMask;
				cvtColor(img, hsv, COLOR_BGR2HSV);
				cvtColor(img, gray, COLOR_BGR2GRAY);
				vector<Mat> hsvCh;
				split(hsv, hsvCh);
				threshold(hsvCh[1], satMask, 180, 255, THRESH_BINARY);
				gray.setTo(255, satMask);
			}
			else
			{
				gray = img.clone();
			}

			// Parameters unchanged from warp_engine.cpp: blockSize=101, C=15.
			Mat binary;
			adaptiveThreshold(gray, binary, 255,
				ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY_INV, 101, 15);

			// Morphological close with L-shaped kernel (do not change — see
			// warp_engine.cpp comment: "各位活爹，上面改的代码和参数调了两节课").
			Mat kernel = getStructuringElement(MORPH_CROSS, Size(2, 2));
			Mat closedBinary;
			morphologyEx(binary, closedBinary, MORPH_CLOSE, kernel);

			vector<vector<Point>> contours;
			vector<Vec4i>         hierarchy;
			findContours(closedBinary, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);

			vector<Marker> centers;
			for (size_t i = 0; i < contours.size(); ++i)
			{
				// Count nesting depth; ≥2 levels = finder pattern.
				int kidIdx = hierarchy[static_cast<int>(i)][2];
				int cnt    = 0;
				while (kidIdx != -1)
				{
					++cnt;
					kidIdx = hierarchy[kidIdx][2];
					if (cnt >= 2) break;
				}
				if (cnt < 2) continue;

				const Moments mu = moments(contours[i]);
				if (mu.m00 >= 500.0)
					centers.push_back({ Point2f(static_cast<float>(mu.m10 / mu.m00),
											static_cast<float>(mu.m01 / mu.m00)),
										static_cast<float>(mu.m00) });
			}

			// Merge duplicates within 100 px; keep the largest-area representative.
			vector<Marker> merged;
			for (const auto& pt : centers)
			{
				bool isNew = true;
				for (auto& m : merged)
				{
					if (norm(pt.center - m.center) < 100.0f)
					{
						if (pt.area > m.area) m = pt;
						isNew = false;
						break;
					}
				}
				if (isNew) merged.push_back(pt);
			}
			return merged;
		}

		// Locate the four perspective corners TL/TR/BR/BL.
		// Algorithm: modify/warp_engine.cpp (FindMarkerCenters + anti-perspective-
		// inversion module + rotation module).
		bool locateCorners(const Mat& srcImg,
			Point2f& tl, Point2f& tr, Point2f& br, Point2f& bl)
		{
			// Pre-scale to ≤800 px (warp_engine.cpp: scale = 800/max(w,h)).
			const float scale = 800.0f / static_cast<float>(std::max(srcImg.cols, srcImg.rows));
			Mat small;
			if (scale < 1.0f)
				resize(srcImg, small, Size(), scale, scale, INTER_AREA);
			else
				small = srcImg;

			vector<Marker> smallMarkers = findMarkers(small);
			if (smallMarkers.size() < 4) return false;

			// Scale marker centres back to original-image coordinates.
			vector<Marker> markers;
			markers.reserve(smallMarkers.size());
			for (const auto& m : smallMarkers)
				markers.push_back({ m.center / scale, m.area });

			// Keep the 4 farthest from the approximate centre (same as warp_engine.cpp).
			Point2f approxCenter(0.0f, 0.0f);
			for (const auto& p : markers) approxCenter += p.center;
			approxCenter /= static_cast<float>(markers.size());

			if (markers.size() > 4)
			{
				std::sort(markers.begin(), markers.end(),
					[&](const Marker& a, const Marker& b)
					{
						return norm(a.center - approxCenter) > norm(b.center - approxCenter);
					});
				markers.resize(4);
			}

			Point2f exactCenter(0.0f, 0.0f);
			for (const auto& p : markers) exactCenter += p.center;
			exactCenter /= 4.0f;

			// ── Anti-perspective-inversion module (warp_engine.cpp) ──────────────
			// BR marker has the smallest area/dist² ratio (it is slightly smaller
			// and farther-equivalent because of perspective foreshortening).
			int   brIdx    = -1;
			float minRatio = 1e9f;
			for (int i = 0; i < 4; ++i)
			{
				const float d     = static_cast<float>(norm(markers[i].center - exactCenter));
				const float ratio = markers[i].area / std::max(d * d, 1.0f);
				if (ratio < minRatio) { minRatio = ratio; brIdx = i; }
			}
			const Point2f brPoint = markers[brIdx].center;

			// ── Rotation module (warp_engine.cpp) ─────────────────────────────────
			// Sort all four centres by angle around exactCenter, then assign
			// br / bl / tl / tr starting from brIdx going counter-clockwise
			// (next = bl, opposite = tl, previous = tr).
			vector<Point2f> sorted;
			sorted.reserve(4);
			for (const auto& m : markers) sorted.push_back(m.center);
			std::sort(sorted.begin(), sorted.end(),
				[&](Point2f a, Point2f b)
				{
					return std::atan2(a.y - exactCenter.y, a.x - exactCenter.x)
						<  std::atan2(b.y - exactCenter.y, b.x - exactCenter.x);
				});

			int sortedBrIdx = 0;
			for (int i = 0; i < 4; ++i)
			{
				if (norm(sorted[i] - brPoint) < 1.0f) { sortedBrIdx = i; break; }
			}

			br = sorted[sortedBrIdx];
			bl = sorted[(sortedBrIdx + 1) % 4];
			tl = sorted[(sortedBrIdx + 2) % 4];
			tr = sorted[(sortedBrIdx + 3) % 4];
			return true;
		}
	}

	bool Main(const Mat& srcImg, Mat& disImg)
	{
		if (srcImg.empty()) return false;

		// Reset cached corners when input resolution changes.
		if (srcImg.cols != g_lastCols || srcImg.rows != g_lastRows)
		{
			g_lastCols = srcImg.cols;
			g_lastRows = srcImg.rows;
			g_frameCount = 0;
			g_cachedCorners.valid = false;
		}

		// Square-input fast path: INTER_AREA downsample to 266×266 (per-pixel vote),
		// then data-only Otsu binarize and write pixel-by-pixel.
		const double aspect = static_cast<double>(srcImg.cols) / srcImg.rows;
		if (aspect > 0.95 && aspect < 1.05 && srcImg.cols > 200)
		{
			Mat imgGray;
			if (srcImg.channels() == 3) cvtColor(srcImg, imgGray, COLOR_BGR2GRAY);
			else                        imgGray = srcImg.clone();

			// INTER_AREA: each output pixel is the area-weighted average of every
			// source pixel that maps to it — the true per-pixel proportional vote.
			Mat avg266;
			resize(imgGray, avg266, Size(kFrameSize, kFrameSize), 0, 0, INTER_AREA);

			const double thresh = dataOtsuThresh(avg266);
			applyPixelBinary(avg266, thresh, disImg);
			return true;
		}

		// Warmup: skip the first few frames until detection is stable.
		if (g_frameCount < 3)
		{
			++g_frameCount;
			return false;
		}

		// Post-localization correction pipeline:
		// 1. Locate four corners via warp_engine.cpp algorithm (UNTOUCHED).
		// 2. warpPerspective (INTER_LINEAR) → targetSize × targetSize grayscale.
		// 2a. INTER_AREA resize → largest 266-multiple ≤ targetSize (integer cell coverage).
		// 2b. INTER_AREA resize → 266×266 (integer ratio, clean per-pixel vote).
		// 3. CLAHE (clipLimit=2, tileGrid=8×8) — normalises local contrast so that
		//    coloured finder patterns and illumination gradients do not bias binarisation.
		// 4. adaptiveThreshold (GAUSSIAN_C, BINARY, blockSize=7, C=3) — each pixel is
		//    classified relative to its local 7×7 Gaussian mean, correctly binarising
		//    both coloured markers and standard black/white data cells.
		// 5. cvtColor(GRAY→BGR) → 266×266 CV_8UC3 output for ImageDecode::Main.
		auto warpBW = [&](Point2f tl, Point2f tr, Point2f br, Point2f bl) -> bool
			{
				Mat grayFull;
				if (srcImg.channels() == 3) cvtColor(srcImg, grayFull, COLOR_BGR2GRAY);
				else                        grayFull = srcImg.clone();

				// Estimate natural barcode pixel size from the finder-centre span.
				// In the warp_engine.cpp mapping: finder span = targetSize*(1-2*0.05225).
				// → targetSize = edgeLen / (1 - 2*0.05225) ≈ edgeLen / 0.8955.
				const float edgeLen = static_cast<float>(
					std::min(norm(tr - tl), norm(bl - tl)));
				const int targetSize = std::max(kFrameSize,
					static_cast<int>(std::round(edgeLen / 0.8955f)));

				// White-border-elimination + direction-correction fractions
				// from warp_engine.cpp (hand-tuned, do not change).
				const float padX    = targetSize * 0.05225f;
				const float padY    = targetSize * 0.05225f;
				const float correct = targetSize * 0.0160f;

				const vector<Point2f> srcPts = { tl, tr, br, bl };
				const vector<Point2f> dstPts = {
					Point2f(padX,                               padY),
					Point2f(targetSize - 1.0f - padX,           padY),
					Point2f(targetSize - 1.0f - padX - correct, targetSize - 1.0f - padY - correct),
					Point2f(padX,                               targetSize - 1.0f - padY)
				};
				const Mat M = getPerspectiveTransform(srcPts, dstPts);

				// Step 1: warp to natural barcode size.
				Mat warped;
				warpPerspective(grayFull, warped, M, Size(targetSize, targetSize), INTER_LINEAR);

				// Step 2a: INTER_AREA resize to the largest multiple of 266 that is
				// ≤ targetSize so every cell covers an integer number of warped pixels.
				Mat intermediate;
				const int scale  = std::max(1, targetSize / kFrameSize);
				const int midSize = scale * kFrameSize;
				if (midSize < targetSize)
					resize(warped, intermediate, Size(midSize, midSize), 0, 0, INTER_AREA);
				else
					intermediate = warped;

				// Step 2b: INTER_AREA final downsample → per-pixel vote at 266×266.
				Mat avg266;
				resize(intermediate, avg266, Size(kFrameSize, kFrameSize), 0, 0, INTER_AREA);

				// Step 3: CLAHE — locally normalises contrast so that coloured finder
				// patterns and uneven camera lighting don't bias a global threshold.
				// tileGridSize 8×8 gives tiles of ~33×33 px ≈ 16 cells each, which
				// is large enough to capture the local background level while still
				// accommodating the illumination gradient across the warped image.
				Ptr<CLAHE> clahe = createCLAHE(2.0, Size(8, 8));
				Mat claheOut;
				clahe->apply(avg266, claheOut);

				// Step 4: Local adaptive binarisation.
				// Each barcode cell occupies 2×2 px in the 266×266 image, so a
				// 7-px Gaussian window covers ~3.5 cells — enough to estimate the
				// local background without blurring across distant illumination zones.
				// THRESH_BINARY: pixel > (gaussian_mean − C) → 255 (white/background),
				//                otherwise → 0   (black/data or finder pattern).
				Mat bin1ch;
				adaptiveThreshold(claheOut, bin1ch, 255,
					ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 7, 3);

				// Step 5: Emit as 266×266 CV_8UC3 for ImageDecode::Main.
				cvtColor(bin1ch, disImg, COLOR_GRAY2BGR);
				return true;
			};

		// Try detection using the warp_engine.cpp algorithm.
		{
			Point2f tl, tr, br, bl;
			if (locateCorners(srcImg, tl, tr, br, bl))
			{
				g_cachedCorners = { tl, tr, br, bl, true };
				return warpBW(tl, tr, br, bl);
			}
		}

		// Fall back to cached corners from the previous successful frame.
		if (g_cachedCorners.valid)
			return warpBW(g_cachedCorners.tl, g_cachedCorners.tr,
				g_cachedCorners.br, g_cachedCorners.bl);

		return false;
	}
} // namespace ImgParse
