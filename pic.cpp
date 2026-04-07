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

		// Square-input fast path: resize with INTER_AREA (proportional voting over source
		// pixels per output cell), then Otsu binarize the averaged 266×266 result.
		const double aspect = static_cast<double>(srcImg.cols) / srcImg.rows;
		if (aspect > 0.95 && aspect < 1.05 && srcImg.cols > 200)
		{
			Mat imgGray;
			if (srcImg.channels() == 3) cvtColor(srcImg, imgGray, COLOR_BGR2GRAY);
			else                        imgGray = srcImg.clone();

			// INTER_AREA averages all source pixels that map into each output cell —
			// this is the proportional majority vote on the original image.
			Mat resized266;
			resize(imgGray, resized266, Size(kFrameSize, kFrameSize), 0, 0, INTER_AREA);

			// After averaging, the histogram is cleanly bimodal; Otsu is reliable here.
			Mat bin266;
			threshold(resized266, bin266, 0, 255, THRESH_BINARY | THRESH_OTSU);

			cvtColor(bin266, disImg, COLOR_GRAY2BGR);
			return true;
		}

		// Warmup: skip the first few frames until detection is stable.
		if (g_frameCount < 3)
		{
			++g_frameCount;
			return false;
		}

		// Forward-warp approach (reference: pic_color.cpp):
		// 1. Warp original grayscale to the barcode's natural pixel size (targetSize).
		// 2. Resize to 266×266 with INTER_AREA — true proportional voting: each output
		//    cell averages every source pixel in its corresponding region, avoiding moiré.
		// 3. Otsu threshold → output as CV_8UC3.
		auto warpBW = [&](Point2f tl, Point2f tr, Point2f br, Point2f bl) -> bool
			{
				Mat grayFull;
				if (srcImg.channels() == 3) cvtColor(srcImg, grayFull, COLOR_BGR2GRAY);
				else                        grayFull = srcImg.clone();

				// Estimate natural barcode side length in source pixels.
				// Finder-marker centres span 224 pixels of the 266-wide full barcode
				// (positions 21 and 245), i.e. 112/133 cells.  Multiply by 133/112.
				const double edgeLen = std::min(norm(tr - tl), norm(bl - tl));
				const int targetSize = std::max(kFrameSize,
					static_cast<int>(std::round(edgeLen * 133.0 / 112.0)));

				const float k = static_cast<float>(targetSize) / kFrameSize;

				// Build transform: finder-marker centres → scaled positions in targetSize space.
				const vector<Point2f> srcPts = { tl, tr, br, bl };
				const vector<Point2f> dstPts = {
					Point2f(21.0f * k, 21.0f * k),
					Point2f(245.0f * k, 21.0f * k),
					Point2f(253.5f * k, 253.5f * k),
					Point2f(21.0f * k, 245.0f * k)
				};
				const Mat M = getPerspectiveTransform(srcPts, dstPts);

				// Step 1: forward-warp to natural barcode size.
				Mat warped;
				warpPerspective(grayFull, warped, M, Size(targetSize, targetSize), INTER_LINEAR);

				// Step 2: resize to 266×266 with INTER_AREA (proportional voting).
				Mat resized266;
				resize(warped, resized266, Size(kFrameSize, kFrameSize), 0, 0, INTER_AREA);

				// Step 3: Otsu threshold → output as CV_8UC3.
				Mat bin;
				threshold(resized266, bin, 0, 255, THRESH_BINARY | THRESH_OTSU);

				cvtColor(bin, disImg, COLOR_GRAY2BGR);
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
