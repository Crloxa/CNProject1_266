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
		constexpr int   kFrameSize        = 266;
		constexpr int   kGridSize         = 266;
		constexpr int   kQuietZone        = 0;
		constexpr int   kLargeFinder      = 42;
		// r_min = (quiet_zone + large_finder/2) / (grid_size + 2*quiet_zone) = 21/266
		// r_max = (logic_total_width - center_offset) / logic_total_width    = 245/266
		constexpr float kRMin             = (kQuietZone + kLargeFinder / 2.0f) / (kGridSize + 2.0f * kQuietZone);
		constexpr float kRMax             = 1.0f - kRMin;
		constexpr int   kThresholdBlock   = 19;
		constexpr int   kThresholdBias    = 10;

		struct Marker
		{
			Point2f center;
			double  area;
		};

		Mat g_lastValidTransform;
		int g_lastCols     = 0;
		int g_lastRows     = 0;
		int g_frameCount   = 0;

		// Find the child contour of parentIdx with the largest area (from modify/pic.cpp).
		int findLargestChild(int parentIdx,
		                     const vector<vector<Point>>& contours,
		                     const vector<Vec4i>& hierarchy)
		{
			int    maxIdx  = -1;
			double maxArea = -1.0;
			int    child   = hierarchy[parentIdx][2];
			while (child >= 0)
			{
				const double a = contourArea(contours[child]);
				if (a > maxArea) { maxArea = a; maxIdx = child; }
				child = hierarchy[child][0];
			}
			return maxIdx;
		}

		// Block-wise RGB-max adaptive threshold binarisation (from modify/pic.cpp).
		void blockwiseColorMaxAdaptiveThreshold(const Mat& imgColor, Mat& binImg)
		{
			const int H       = imgColor.rows;
			const int W       = imgColor.cols;
			const int nBlockY = (H + kThresholdBlock - 1) / kThresholdBlock;
			const int nBlockX = (W + kThresholdBlock - 1) / kThresholdBlock;
			vector<vector<int>> thresholds(nBlockY, vector<int>(nBlockX, 128));

			for (int by = 0; by < nBlockY; ++by)
			{
				for (int bx = 0; bx < nBlockX; ++bx)
				{
					const int y0 = by * kThresholdBlock;
					const int y1 = std::min(y0 + kThresholdBlock, H);
					const int x0 = bx * kThresholdBlock;
					const int x1 = std::min(x0 + kThresholdBlock, W);
					vector<int> values;
					values.reserve((y1 - y0) * (x1 - x0));
					for (int y = y0; y < y1; ++y)
						for (int x = x0; x < x1; ++x)
						{
							const Vec3b pix = imgColor.at<Vec3b>(y, x);
							values.push_back(std::max(pix[0], std::max(pix[1], pix[2])));
						}
					if (!values.empty())
					{
						sort(values.begin(), values.end());
						const int n       = static_cast<int>(values.size());
						const int lowIdx  = n / 10;
						const int highIdx = n - n / 10 - 1;
						int thres = (values[lowIdx] + values[highIdx]) / 2 + kThresholdBias;
						thres = std::max(0, std::min(255, thres));
						thresholds[by][bx] = thres;
					}
				}
			}

			binImg.create(H, W, CV_8UC3);
			for (int y = 0; y < H; ++y)
			{
				const int by = y / kThresholdBlock;
				for (int x = 0; x < W; ++x)
				{
					const int bx  = x / kThresholdBlock;
					const Vec3b p = imgColor.at<Vec3b>(y, x);
					const int   mx = std::max(p[0], std::max(p[1], p[2]));
					binImg.at<Vec3b>(y, x) = (mx > thresholds[by][bx])
					                         ? Vec3b(255, 255, 255) : Vec3b(0, 0, 0);
				}
			}
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

			const double minArea    = std::max(15.0, static_cast<double>(maxDim) * maxDim * 0.000004);
			const double mergeDist  = std::max(15.0, static_cast<double>(maxDim) * 0.0078);

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
			double maxDist   = 0.0;
			int    rightAngleIdx = -1;
			for (int i = 0; i < 3; ++i)
				for (int j = i + 1; j < 3; ++j)
				{
					const double d = norm(markers[i].center - markers[j].center);
					if (d > maxDist) { maxDist = d; rightAngleIdx = 3 - i - j; }
				}

			const Point2f tlCand = markers[rightAngleIdx].center;
			const Point2f pt1    = markers[(rightAngleIdx + 1) % 3].center;
			const Point2f pt2    = markers[(rightAngleIdx + 2) % 3].center;

			const Point2f v1   = pt1 - tlCand;
			const Point2f v2   = pt2 - tlCand;
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
			else           { trCand = pt2; blCand = pt1; }

			// BR: use the detected 4th marker if close to the parallelogram prediction,
			// otherwise estimate it.
			Point2f brCand     = trCand + blCand - tlCand;
			const double maxLeg = std::max(len1, len2);
			if (markers.size() > 3)
			{
				double minD    = 1e9;
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
			g_lastCols   = srcImg.cols;
			g_lastRows   = srcImg.rows;
			g_frameCount = 0;
			g_lastValidTransform = Mat();
		}

		// Warmup: skip the first few frames until detection is stable (from modify/pic.cpp).
		if (g_frameCount < 3)
		{
			++g_frameCount;
			return false;
		}

		auto warpAndThreshold = [&](const Mat& M) -> bool
		{
			Mat warped;
			warpPerspective(srcImg, warped, M, Size(kFrameSize, kFrameSize), INTER_NEAREST);
			blockwiseColorMaxAdaptiveThreshold(warped, disImg);
			return true;
		};

		auto useCached = [&]() -> bool
		{
			if (g_lastValidTransform.empty()) return false;
			return warpAndThreshold(g_lastValidTransform);
		};

		// Try detection without HSV assist, then with (from modify/pic.cpp).
		const int passes = (srcImg.channels() == 3) ? 2 : 1;
		for (int pass = 0; pass < passes; ++pass)
		{
			Point2f tl, tr, br, bl;
			if (!locateCorners(srcImg, pass == 1, tl, tr, br, bl))
				continue;

			// Build perspective transform using the r_min/r_max formula.
			const vector<Point2f> srcPts = { tl, tr, br, bl };
			const vector<Point2f> dstPts = {
				Point2f(kRMin * kFrameSize, kRMin * kFrameSize),
				Point2f(kRMax * kFrameSize, kRMin * kFrameSize),
				Point2f(kRMax * kFrameSize, kRMax * kFrameSize),
				Point2f(kRMin * kFrameSize, kRMax * kFrameSize)
			};
			const Mat M = getPerspectiveTransform(srcPts, dstPts);
			g_lastValidTransform = M.clone();
			return warpAndThreshold(M);
		}

		return useCached();
	}
} // namespace ImgParse
