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

		Mat g_lastValidTransform;
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

		// Reset cached transform when input resolution changes (from modify/pic.cpp).
		if (srcImg.cols != g_lastCols || srcImg.rows != g_lastRows)
		{
			g_lastCols = srcImg.cols;
			g_lastRows = srcImg.rows;
			g_frameCount = 0;
			g_lastValidTransform = Mat();
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

		// Warmup: skip the first few frames until detection is stable (from modify/pic.cpp).
		if (g_frameCount < 3)
		{
			++g_frameCount;
			return false;
		}

		auto warpColor = [&](const Mat& M) -> bool
			{
				// Step 1: convert to grayscale (work on the original, unprocessed image).
				Mat grayFull;
				if (srcImg.channels() == 3) cvtColor(srcImg, grayFull, COLOR_BGR2GRAY);
				else                        grayFull = srcImg.clone();

				// Step 2: compute inverse perspective transform so we can map each
				//         output cell back to its location in the original image.
				Mat M_inv;
				invert(M, M_inv);

				// Helper: map an output point (x, y) to source coordinates.
				auto mapPt = [&](float x, float y) -> Point2f
					{
						const double wx = M_inv.at<double>(0, 0) * x + M_inv.at<double>(0, 1) * y + M_inv.at<double>(0, 2);
						const double wy = M_inv.at<double>(1, 0) * x + M_inv.at<double>(1, 1) * y + M_inv.at<double>(1, 2);
						const double wz = M_inv.at<double>(2, 0) * x + M_inv.at<double>(2, 1) * y + M_inv.at<double>(2, 2);
						return Point2f(static_cast<float>(wx / wz), static_cast<float>(wy / wz));
					};

				// Step 3: estimate the local scale (source pixels per output cell) at the
				//         image centre so we know how large a patch to sample per cell.
				const float fc = kFrameSize / 2.0f, fr = kFrameSize / 2.0f;
				const float scale = static_cast<float>(norm(mapPt(fc + 1.0f, fr) - mapPt(fc, fr)));
				const int halfR = std::max(1, static_cast<int>(scale * 0.5f));

				// Step 4: determine binarisation threshold from the barcode region only.
				//         Warp the grayscale with INTER_LINEAR to get a 266×266 view of
				//         just the barcode, then let Otsu pick the split on that bimodal
				//         histogram.  This avoids the threshold being skewed by background
				//         pixels outside the barcode.
				Mat warpedForThresh;
				warpPerspective(grayFull, warpedForThresh, M, Size(kFrameSize, kFrameSize), INTER_LINEAR);
				Mat tmpBin;
				const double otsuThresh = threshold(warpedForThresh, tmpBin, 0, 255, THRESH_BINARY | THRESH_OTSU);

				// Step 5: for each output cell, vote proportionally on the original image.
				//         Each cell maps back to a patch of roughly (2*halfR+1)² source
				//         pixels.  We count how many exceed the Otsu threshold and declare
				//         the cell white if the majority does — true proportional voting
				//         directly on the original, without any prior binarisation step.
				disImg.create(kFrameSize, kFrameSize, CV_8UC3);
				const int srcCols = grayFull.cols, srcRows = grayFull.rows;
				for (int r = 0; r < kFrameSize; ++r)
					for (int c = 0; c < kFrameSize; ++c)
					{
						const Point2f sp = mapPt(static_cast<float>(c), static_cast<float>(r));
						const int cx0 = static_cast<int>(sp.x + 0.5f);
						const int cy0 = static_cast<int>(sp.y + 0.5f);
						int white = 0, total = 0;
						for (int dy = -halfR; dy <= halfR; ++dy)
							for (int dx = -halfR; dx <= halfR; ++dx)
							{
								const int px = std::max(0, std::min(srcCols - 1, cx0 + dx));
								const int py = std::max(0, std::min(srcRows - 1, cy0 + dy));
								if (grayFull.at<uint8_t>(py, px) > static_cast<uint8_t>(otsuThresh))
									++white;
								++total;
							}
						disImg.at<Vec3b>(r, c) = (white * 2 >= total)
							? Vec3b(255, 255, 255) : Vec3b(0, 0, 0);
					}
				return true;
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
			const vector<Point2f> dstPts = {
				Point2f(21.0f, 21.0f),
				Point2f(245.0f, 21.0f),
				Point2f(253.5f, 253.5f),
				Point2f(21.0f, 245.0f)
			};
			const Mat M = getPerspectiveTransform(srcPts, dstPts);
			g_lastValidTransform = M.clone();
			return warpColor(M);
		}

		return useCached();
	}
} // namespace ImgParse
