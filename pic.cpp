#include "pic.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace ImgParse
{
	using namespace cv;
	using namespace std;

	namespace
	{
		constexpr int   kFrameSize = 266;
		// CLAHE settings for moderate local contrast boost without halo artifacts.
		constexpr double kClaheClipLimit = 2.0;
		constexpr int   kClaheGridSize = 8;
		// Light denoise before global threshold to suppress isolated sensor noise.
		constexpr int   kMedianKernel = 3;
		// Marker extraction and corner ordering thresholds from warp_engine.cpp.
		constexpr double kMinMarkerMomentArea = 500.0;
		constexpr double kMarkerMergeDistance = 100.0;
		constexpr float kPointMatchTolerance = 1.0f;
		constexpr float kMaxScaledDimension = 800.0f;
		constexpr int kLocateAdaptiveBlockSize = 101;
		constexpr int kLocateAdaptiveC = 15;
		constexpr int kLocateMorphKernel = 2;
		constexpr double kDistanceEpsilon = 1e-6;
		// Perspective correction fractions from warp_engine.cpp.
		constexpr float kPadFraction = 0.05225f;
		constexpr float kCorrectFraction = 0.0160f;

		struct Marker
		{
			Point2f center;
			double  area;
		};

		Mat g_lastValidTransform;
		int g_lastCols = 0;
		int g_lastRows = 0;
		int g_frameCount = 0;

		bool buildBinaryOutput(const Mat& src, Mat& disImg)
		{
			if (src.empty()) return false;

			Mat gray;
			if (src.channels() == 3) cvtColor(src, gray, COLOR_BGR2GRAY);
			else                     gray = src;

			Mat contrast;
			Ptr<CLAHE> clahe = createCLAHE(kClaheClipLimit, Size(kClaheGridSize, kClaheGridSize));
			clahe->apply(gray, contrast);

			Mat denoise;
			medianBlur(contrast, denoise, kMedianKernel);

			Mat binRaw;
			// Convert to a binary-like decode frame (266x266 BGR) with light local
			// contrast enhancement and denoise before Otsu thresholding.
			threshold(denoise, binRaw, 0, 255, THRESH_BINARY | THRESH_OTSU);
			if (binRaw.empty()) return false;
			if (binRaw.rows != kFrameSize || binRaw.cols != kFrameSize) return false;

			cvtColor(binRaw, disImg, COLOR_GRAY2BGR);
			return !disImg.empty()
				&& disImg.rows == kFrameSize
				&& disImg.cols == kFrameSize
				&& disImg.type() == CV_8UC3;
		}

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

		// Locate the four perspective corners TL/TR/BR/BL using the same anti-inversion
		// ordering strategy as modify/warp_engine.cpp.
		bool locateCorners(const Mat& srcImg, bool useHSV,
			Point2f& tl, Point2f& tr, Point2f& br, Point2f& bl)
		{
			if (srcImg.empty()) return false;
			Mat gray;
			if (srcImg.channels() == 3) cvtColor(srcImg, gray, COLOR_BGR2GRAY);
			else                        gray = srcImg.clone();

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

			float scale = kMaxScaledDimension / static_cast<float>(std::max(srcImg.cols, srcImg.rows));
			if (scale > 1.0f) scale = 1.0f;
			Mat smallImg;
			resize(gray, smallImg, Size(), scale, scale, INTER_AREA);

			Mat binary;
			adaptiveThreshold(smallImg, binary, 255,
				ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY_INV, kLocateAdaptiveBlockSize, kLocateAdaptiveC);

			Mat kernel = getStructuringElement(MORPH_CROSS, Size(kLocateMorphKernel, kLocateMorphKernel));
			Mat closedBinary;
			morphologyEx(binary, closedBinary, MORPH_CLOSE, kernel);

			vector<vector<Point>> contours;
			vector<Vec4i>         hierarchy;
			findContours(closedBinary, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);

			vector<Marker> markers;
			for (size_t i = 0; i < contours.size(); ++i)
			{
				int child = hierarchy[i][2];
				int depth = 0;
				while (child != -1)
				{
					++depth;
					child = hierarchy[child][2];
					if (depth >= 2) break;
				}
				if (depth < 2) continue;
				const Moments mu = moments(contours[i], false);
				if (mu.m00 < kMinMarkerMomentArea) continue;
				markers.push_back({
					Point2f(static_cast<float>(mu.m10 / mu.m00 / scale),
							static_cast<float>(mu.m01 / mu.m00 / scale)),
					mu.m00 / (scale * scale)
					});
			}

			// Merge duplicates using warp_engine distance threshold.
			vector<Marker> unique;
			for (const auto& m : markers)
			{
				bool dup = false;
				for (auto& u : unique)
				{
					if (norm(m.center - u.center) < kMarkerMergeDistance)
					{
						if (m.area > u.area) u = m;
						dup = true;
						break;
					}
				}
				if (!dup) unique.push_back(m);
			}
			markers = unique;

			if (markers.size() < 4) return false;

			Point2f approxCenter(0.0f, 0.0f);
			for (const auto& m : markers) approxCenter += m.center;
			approxCenter *= (1.0f / static_cast<float>(markers.size()));
			if (markers.size() > 4)
			{
				std::sort(markers.begin(), markers.end(), [&](const Marker& a, const Marker& b) {
					return norm(a.center - approxCenter) > norm(b.center - approxCenter);
					});
				markers.resize(4);
			}

			Point2f exactCenter(0.0f, 0.0f);
			for (const auto& m : markers) exactCenter += m.center;
			exactCenter *= 0.25f;

			int brIdx = -1;
			double minRatio = std::numeric_limits<double>::max();
			for (int i = 0; i < 4; ++i)
			{
				const double dist = norm(markers[i].center - exactCenter);
				if (dist < kDistanceEpsilon) continue;
				const double ratio = markers[i].area / (dist * dist);
				if (ratio < minRatio)
				{
					minRatio = ratio;
					brIdx = i;
				}
			}
			if (brIdx < 0) return false;

			const Point2f brPoint = markers[brIdx].center;
			vector<Point2f> sortedPts;
			sortedPts.reserve(4);
			for (const auto& m : markers) sortedPts.push_back(m.center);
			std::sort(sortedPts.begin(), sortedPts.end(), [&](const Point2f& a, const Point2f& b) {
				return std::atan2(a.y - exactCenter.y, a.x - exactCenter.x)
					< std::atan2(b.y - exactCenter.y, b.x - exactCenter.x);
				});

			int sortedBrIdx = -1;
			for (int i = 0; i < 4; ++i)
			{
				if (norm(sortedPts[i] - brPoint) < kPointMatchTolerance)
				{
					sortedBrIdx = i;
					break;
				}
			}
			if (sortedBrIdx < 0) return false;

			br = sortedPts[sortedBrIdx];
			bl = sortedPts[(sortedBrIdx + 1) % 4];
			tl = sortedPts[(sortedBrIdx + 2) % 4];
			tr = sortedPts[(sortedBrIdx + 3) % 4];
			return true;
		}
	}

	bool Main(const Mat& srcImg, Mat& disImg)
	{
		if (srcImg.empty()) return false;

		// Reset cached transform when input resolution changes (from modify/pic.cpp).
		if (srcImg.cols != g_lastCols || srcImg.rows != g_lastRows)
		{
			g_lastCols = srcImg.cols;
			g_lastRows = srcImg.rows;
			g_frameCount = 0;
			g_lastValidTransform = Mat();
		}

		// Square-input fast path: direct Otsu resize (from modify/pic.cpp).
		const double aspect = static_cast<double>(srcImg.cols) / srcImg.rows;
		if (aspect > 0.95 && aspect < 1.05 && srcImg.cols > 200)
		{
			Mat imgGray;
			if (srcImg.channels() == 3) cvtColor(srcImg, imgGray, COLOR_BGR2GRAY);
			else                        imgGray = srcImg.clone();
			Mat binRaw;
			threshold(imgGray, binRaw, 0, 255, THRESH_BINARY | THRESH_OTSU);
			disImg.create(kFrameSize, kFrameSize, CV_8UC3);
			const float stepX = static_cast<float>(srcImg.cols) / kFrameSize;
			const float stepY = static_cast<float>(srcImg.rows) / kFrameSize;
			for (int r = 0; r < kFrameSize; ++r)
				for (int c = 0; c < kFrameSize; ++c)
				{
					const int px = std::min(static_cast<int>((c + 0.5f) * stepX), srcImg.cols - 1);
					const int py = std::min(static_cast<int>((r + 0.5f) * stepY), srcImg.rows - 1);
					const uint8_t val = binRaw.at<uint8_t>(py, px);
					disImg.at<Vec3b>(r, c) = val ? Vec3b(255, 255, 255) : Vec3b(0, 0, 0);
				}
			return true;
		}

		// Warmup: skip the first few frames until detection is stable (from modify/pic.cpp).
		if (g_frameCount < 3)
		{
			++g_frameCount;
			return false;
		}

		auto warpColor = [&](const Mat& M) -> bool
			{
				Mat warped;
				// Keep module boundaries crisp for downstream binary sampling.
				warpPerspective(srcImg, warped, M, Size(kFrameSize, kFrameSize), INTER_NEAREST);
				return buildBinaryOutput(warped, disImg);
			};

		auto useCached = [&]() -> bool
			{
				if (g_lastValidTransform.empty()) return false;
				return warpColor(g_lastValidTransform);
			};

		// Try detection without HSV assist, then with (from modify/pic.cpp).
		const int passes = (srcImg.channels() == 3) ? 2 : 1;
		for (int pass = 0; pass < passes; ++pass)
		{
			Point2f tl, tr, br, bl;
			if (!locateCorners(srcImg, pass == 1, tl, tr, br, bl))
				continue;

			// Build perspective transform using the corrected dst points (per warp_engine.cpp convention).
			const vector<Point2f> srcPts = { tl, tr, br, bl };
			const float padX = kFrameSize * kPadFraction;
			const float padY = kFrameSize * kPadFraction;
			const float correct = kFrameSize * kCorrectFraction;
			const vector<Point2f> dstPts = {
				Point2f(padX, padY),
				Point2f(kFrameSize - 1.0f - padX, padY),
				Point2f(kFrameSize - 1.0f - padX - correct, kFrameSize - 1.0f - padY - correct),
				Point2f(padX, kFrameSize - 1.0f - padY)
			};
			const Mat M = getPerspectiveTransform(srcPts, dstPts);
			g_lastValidTransform = M.clone();
			return warpColor(M);
		}

		return useCached();
	}
} // namespace ImgParse
