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
			double  area;
		};

		// Cached corners from the most recent successful detection.
		struct CachedCorners { Point2f tl, tr, br, bl; bool valid = false; };
		CachedCorners g_cachedCorners;
		int g_lastCols = 0;
		int g_lastRows = 0;
		int g_frameCount = 0;

		// Find the child contour of parentIdx with the largest area (from modify/pic.cpp).
		int findLargestChild(int parentIdx,
			const vector<vector<Point>>& contours,
			const vector<Vec4i>& hierarchy)
		{
			int    maxIdx = -1;
			double maxArea = -1.0;
			int    child = hierarchy[parentIdx][2];
			while (child >= 0)
			{
				const double a = contourArea(contours[child]);
				if (a > maxArea) { maxArea = a; maxIdx = child; }
				child = hierarchy[child][0];
			}
			return maxIdx;
		}

		// Locate the four perspective corners TL/TR/BR/BL using the processV15 approach
		// from modify/pic.cpp (3-ring nested contour detection + geometry validation).
		// useHSV: also suppress saturated-colour regions (same as modify/pic.cpp).
		bool locateCorners(const Mat& srcImg, bool useHSV,
			Point2f& tl, Point2f& tr, Point2f& br, Point2f& bl)
		{
			Mat gray;
			if (srcImg.channels() == 3)
				cvtColor(srcImg, gray, COLOR_BGR2GRAY);
			else
				gray = srcImg.clone();

			if (useHSV && srcImg.channels() == 3)
			{
				Mat hsv;
				cvtColor(srcImg, hsv, COLOR_BGR2HSV);
				vector<Mat> hsvCh;
				split(hsv, hsvCh);
				Mat satMask;
				threshold(hsvCh[1], satMask, 180, 255, THRESH_BINARY);
				gray.setTo(255, satMask);
			}

			const int maxDim = std::max(srcImg.cols, srcImg.rows);

			int blurSz = std::max(5, static_cast<int>(maxDim * 7.0 / 1920.0));
			if (blurSz % 2 == 0) ++blurSz;
			Mat blurred;
			GaussianBlur(gray, blurred, Size(blurSz, blurSz), 0);

			int blockSz = std::max(31, static_cast<int>(maxDim * 31.0 / 1920.0));
			if (blockSz % 2 == 0) ++blockSz;
			Mat binary;
			adaptiveThreshold(blurred, binary, 255,
				ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY_INV, blockSz, 10);

			Mat kernel = getStructuringElement(MORPH_CROSS, Size(2, 2));
			Mat closedBinary;
			morphologyEx(binary, closedBinary, MORPH_CLOSE, kernel);

			vector<vector<Point>> contours;
			vector<Vec4i>         hierarchy;
			findContours(closedBinary, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);

			const double minArea = std::max(15.0, static_cast<double>(maxDim) * maxDim * 0.000004);
			const double mergeDist = std::max(15.0, static_cast<double>(maxDim) * 0.0078);

			vector<Marker> markers;
			for (size_t i = 0; i < contours.size(); ++i)
			{
				const int c1 = findLargestChild(static_cast<int>(i), contours, hierarchy);
				if (c1 < 0) continue;
				const int c2 = findLargestChild(c1, contours, hierarchy);
				if (c2 < 0) continue;
				const double a0 = contourArea(contours[i]);
				const double a1 = contourArea(contours[c1]);
				const double a2 = contourArea(contours[c2]);
				if (a0 < minArea) continue;
				const double r01 = a0 / std::max(a1, 1.0);
				const double r12 = a1 / std::max(a2, 1.0);
				if (r01 > 1.2 && r01 < 8.0 && r12 > 1.2 && r12 < 8.0)
				{
					const Moments mu = moments(contours[i]);
					if (mu.m00 != 0)
						markers.push_back({ Point2f(static_cast<float>(mu.m10 / mu.m00),
													static_cast<float>(mu.m01 / mu.m00)), a0 });
				}
			}

			// Merge duplicates — keep the largest-area representative.
			vector<Marker> unique;
			for (const auto& m : markers)
			{
				bool dup = false;
				for (auto& u : unique)
				{
					if (norm(m.center - u.center) < mergeDist)
					{
						if (m.area > u.area) u = m;
						dup = true;
						break;
					}
				}
				if (!dup) unique.push_back(m);
			}
			markers = unique;

			if (markers.size() < 3) return false;

			// Sort by area descending; the three largest are the structural finders.
			std::sort(markers.begin(), markers.end(),
				[](const Marker& a, const Marker& b) { return a.area > b.area; });

			// The marker farthest from its two neighbours (i.e. at the right-angle corner) is TL.
			double maxDist = 0.0;
			int    rightAngleIdx = -1;
			for (int i = 0; i < 3; ++i)
				for (int j = i + 1; j < 3; ++j)
				{
					const double d = norm(markers[i].center - markers[j].center);
					if (d > maxDist) { maxDist = d; rightAngleIdx = 3 - i - j; }
				}

			const Point2f tlCand = markers[rightAngleIdx].center;
			const Point2f pt1 = markers[(rightAngleIdx + 1) % 3].center;
			const Point2f pt2 = markers[(rightAngleIdx + 2) % 3].center;

			const Point2f v1 = pt1 - tlCand;
			const Point2f v2 = pt2 - tlCand;
			const double  len1 = norm(v1);
			const double  len2 = norm(v2);

			// Validate near-right-angle geometry.
			const double legRatio = len1 / std::max(len2, 1.0);
			if (legRatio < 0.4 || legRatio > 2.5) return false;
			const double cosTheta = (v1.x * v2.x + v1.y * v2.y) / std::max(len1 * len2, 1.0);
			if (std::abs(cosTheta) > 0.75) return false;

			// Cross product determines which leg is TR and which is BL.
			const double cross = v1.x * v2.y - v1.y * v2.x;
			Point2f trCand, blCand;
			if (cross > 0) { trCand = pt1; blCand = pt2; }
			else { trCand = pt2; blCand = pt1; }

			// BR: use the detected 4th marker if close to the parallelogram prediction,
			// otherwise estimate it.
			Point2f brCand = trCand + blCand - tlCand;
			const double maxLeg = std::max(len1, len2);
			if (markers.size() > 3)
			{
				double minD = 1e9;
				int    bestIdx = -1;
				for (size_t i = 3; i < markers.size(); ++i)
				{
					const double d = norm(markers[i].center - brCand);
					if (d < minD) { minD = d; bestIdx = static_cast<int>(i); }
				}
				if (bestIdx >= 0 && minD < maxLeg * 0.4)
					brCand = markers[bestIdx].center;
			}

			tl = tlCand;
			tr = trCand;
			br = brCand;
			bl = blCand;
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

		// Grid constants: the barcode is a 133×133 cell grid.
		// Finder-marker centres land at cell 21 (TL/TR/BL) and cell ~253.5 (BR).
		// kFinderSpan is the cell distance between TL and TR/BL finder centres (112 cells).
		constexpr float kMarkerTL    = 21.0f;   // finder centre position (TL, TR, BL axes)
		constexpr float kMarkerTR    = 245.0f;  // finder centre position (TR col, BL row)
		constexpr float kMarkerBR    = 253.5f;  // finder centre position for BR corner
		constexpr int   kBarcodeGrid = 133;     // full grid dimension in cells
		constexpr int   kFinderSpan  = 112;     // finder-centre-to-finder-centre distance

		// Per-pixel voting at 266×266 resolution (matching code.cpp's encoding):
		// 1. Warp original grayscale to a "natural" barcode pixel size (targetSize).
		// 2a. INTER_AREA resize to the largest multiple of 266 ≤ targetSize so that
		//     every intermediate pixel covers an integer number of warped pixels.
		// 2b. INTER_AREA final resize to 266×266 — integer ratio → clean per-pixel vote.
		// 3. Compute Otsu threshold from data-region pixels only (header + 5 data areas +
		//    corner cells) so finder-pattern and safe-area pixels don't bias the histogram.
		// 4. Classify every 266×266 pixel independently as black or white.
		auto warpBW = [&](Point2f tl, Point2f tr, Point2f br, Point2f bl) -> bool
			{
				Mat grayFull;
				if (srcImg.channels() == 3) cvtColor(srcImg, grayFull, COLOR_BGR2GRAY);
				else                        grayFull = srcImg.clone();

				// Estimate natural barcode side length in source pixels.
				const double edgeLen = std::min(norm(tr - tl), norm(bl - tl));
				const int targetSize = std::max(kFrameSize,
					static_cast<int>(std::round(edgeLen * kBarcodeGrid / kFinderSpan)));

				const float k = static_cast<float>(targetSize) / kFrameSize;

				// Map finder centres to their known pixel positions in the 266 frame,
				// scaled up to targetSize.
				const vector<Point2f> srcPts = { tl, tr, br, bl };
				const vector<Point2f> dstPts = {
					Point2f(kMarkerTL * k, kMarkerTL * k),
					Point2f(kMarkerTR * k, kMarkerTL * k),
					Point2f(kMarkerBR * k, kMarkerBR * k),
					Point2f(kMarkerTL * k, kMarkerTR * k)
				};
				const Mat M = getPerspectiveTransform(srcPts, dstPts);

				// Step 1: warp to natural barcode size.
				Mat warped;
				warpPerspective(grayFull, warped, M, Size(targetSize, targetSize), INTER_LINEAR);

				// Step 2a: INTER_AREA resize to the largest multiple of 266 that is
				// ≤ targetSize.  This makes each cell an integer number of pixels, so
				// the subsequent final resize (step 2b) is a clean integer ratio with
				// no fractional-pixel artifacts.
				Mat intermediate;
				const int scale = std::max(1, targetSize / kFrameSize);
				const int midSize = scale * kFrameSize;
				if (midSize < targetSize)
					resize(warped, intermediate, Size(midSize, midSize), 0, 0, INTER_AREA);
				else
					intermediate = warped;

				// Step 2b: INTER_AREA final downsample → per-pixel vote at 266×266.
				Mat avg266;
				resize(intermediate, avg266, Size(kFrameSize, kFrameSize), 0, 0, INTER_AREA);

				// Step 3: data-only Otsu (unbiased by finder patterns / safe area).
				const double thresh = dataOtsuThresh(avg266);

				// Step 4: per-pixel black/white classification.
				applyPixelBinary(avg266, thresh, disImg);
				return true;
			};

		// Try detection without HSV assist, then with.
		const int passes = (srcImg.channels() == 3) ? 2 : 1;
		for (int pass = 0; pass < passes; ++pass)
		{
			Point2f tl, tr, br, bl;
			if (!locateCorners(srcImg, pass == 1, tl, tr, br, bl))
				continue;

			g_cachedCorners = { tl, tr, br, bl, true };
			return warpBW(tl, tr, br, bl);
		}

		// Fall back to cached corners from the previous successful frame.
		if (g_cachedCorners.valid)
			return warpBW(g_cachedCorners.tl, g_cachedCorners.tr,
				g_cachedCorners.br, g_cachedCorners.bl);

		return false;
	}
} // namespace ImgParse
