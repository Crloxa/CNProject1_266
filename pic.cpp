#include "pic.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

namespace ImgParse
{
	using namespace cv;
	using namespace std;

	namespace
	{
		constexpr int kFrameSize = 266;
		constexpr int kGridSize = 266;                         // logical data grid size in output pixels
		constexpr int kQuietZone = 0;                          // quiet zone width in output pixels
		constexpr int kLargeFinder = 42;                       // large finder marker size in output pixels
		// r_min = (quiet_zone + large_finder/2) / (grid_size + 2*quiet_zone) = 21/266
		// r_max = 1 - r_min = 245/266
		// dst corners are symmetric: TL=(r_min,r_min), TR=(r_max,r_min), BR=(r_max,r_max), BL=(r_min,r_max)
		constexpr float kRMin = (kQuietZone + kLargeFinder / 2.0f) / (kGridSize + 2.0f * kQuietZone);
		constexpr float kRMax = 1.0f - kRMin;
		constexpr float kMaxDetectionDimension = 800.0f;       // always resize to this before detection (same as modify/warp_engine)
		constexpr int kDetectAdaptiveBlock = 101;              // tuned for 800px images (matches modify/warp_engine.cpp)
		constexpr double kDetectMinArea = 500.0;               // tuned for 800px images (matches modify/warp_engine.cpp)
		constexpr float kDetectMergeDistance = 100.0f;         // tuned for 800px images (matches modify/warp_engine.cpp)
		constexpr int kThresholdBlockSize = 19;
		constexpr int kThresholdBias = 10;

		struct Marker
		{
			Point2f center;
			float area;
		};

		Mat g_lastValidTransform;

		vector<Marker> findMarkerCenters(const Mat& input)
		{
			Mat gray;
			if (input.channels() == 3)
			{
				Mat hsv;
				cvtColor(input, hsv, COLOR_BGR2HSV);
				cvtColor(input, gray, COLOR_BGR2GRAY);
				vector<Mat> hsvChannels;
				split(hsv, hsvChannels);
				Mat saturationMask;
				threshold(hsvChannels[1], saturationMask, 180, 255, THRESH_BINARY);
				gray.setTo(255, saturationMask);
			}
			else
			{
				gray = input.clone();
			}

			Mat binary;
			adaptiveThreshold(gray, binary, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY_INV, kDetectAdaptiveBlock, 15);
			Mat kernel = getStructuringElement(MORPH_CROSS, Size(2, 2));
			Mat closedBinary;
			morphologyEx(binary, closedBinary, MORPH_CLOSE, kernel);

			vector<vector<Point>> contours;
			vector<Vec4i> hierarchy;
			findContours(closedBinary, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);

			vector<Marker> centers;
			for (size_t i = 0; i < contours.size(); ++i)
			{
				int kid = hierarchy[i][2];
				int nested = 0;
				while (kid != -1)
				{
					++nested;
					kid = hierarchy[kid][2];
					if (nested >= 2)
					{
						break;
					}
				}
				if (nested >= 2)
				{
					const Moments mu = moments(contours[i], false);
					if (mu.m00 >= kDetectMinArea)
					{
						const Point2f markerCenter(
							static_cast<float>(mu.m10 / mu.m00),
							static_cast<float>(mu.m01 / mu.m00)
						);
						centers.push_back({ markerCenter, static_cast<float>(mu.m00) });
					}
				}
			}

			vector<Marker> merged;
			for (const auto& pt : centers)
			{
				bool isNew = true;
				for (auto& m : merged)
				{
					if (norm(pt.center - m.center) < kDetectMergeDistance)
					{
						isNew = false;
						if (pt.area > m.area)
						{
							m = pt;
						}
						break;
					}
				}
				if (isNew)
				{
					merged.push_back(pt);
				}
			}
			return merged;
		}

		void blockwiseColorMaxAdaptiveThreshold(const Mat& imgColor, Mat& binImg)
		{
			const int H = imgColor.rows;
			const int W = imgColor.cols;
			const int nBlockY = (H + kThresholdBlockSize - 1) / kThresholdBlockSize;
			const int nBlockX = (W + kThresholdBlockSize - 1) / kThresholdBlockSize;
			vector<vector<int>> thresholds(nBlockY, vector<int>(nBlockX, 128));

			for (int by = 0; by < nBlockY; ++by)
			{
				for (int bx = 0; bx < nBlockX; ++bx)
				{
					vector<int> values;
					const int y0 = by * kThresholdBlockSize;
					const int y1 = std::min(y0 + kThresholdBlockSize, H);
					const int x0 = bx * kThresholdBlockSize;
					const int x1 = std::min(x0 + kThresholdBlockSize, W);
					values.reserve((y1 - y0) * (x1 - x0));
					for (int y = y0; y < y1; ++y)
					{
						for (int x = x0; x < x1; ++x)
						{
							const Vec3b pix = imgColor.at<Vec3b>(y, x);
							values.push_back(std::max(pix[0], std::max(pix[1], pix[2])));
						}
					}
					if (!values.empty())
					{
						sort(values.begin(), values.end());
						const int n = static_cast<int>(values.size());
						const int lowIdx = n / 10;
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
				const int by = y / kThresholdBlockSize;
				for (int x = 0; x < W; ++x)
				{
					const int bx = x / kThresholdBlockSize;
					const int thres = thresholds[by][bx];
					const Vec3b pix = imgColor.at<Vec3b>(y, x);
					const int mx = std::max(pix[0], std::max(pix[1], pix[2]));
					binImg.at<Vec3b>(y, x) = (mx > thres) ? Vec3b(255, 255, 255) : Vec3b(0, 0, 0);
				}
			}
		}

		// Sort 4 markers into [TL, TR, BR, BL] using the anti-perspective-flip + rotation
		// logic from modify/warp_engine.cpp.
		// Requires at least 4 markers. If more than 4 are present the 4 farthest from the
		// approximate center are kept (same selection as warp_engine).
		// BR is identified by the minimum area/(dist²) ratio — the small BR finder has a
		// disproportionately small area relative to its distance from the center compared to
		// the three large corner finders.  The remaining three are then assigned by sorting
		// all four by polar angle around the exact centroid.
		bool sortFourMarkers(const vector<Marker>& markers, array<Point2f, 4>& ordered)
		{
			if (markers.size() < 4)
			{
				return false;
			}

			vector<Marker> pts(markers);

			// Approximate center over all detected candidates.
			Point2f approxCenter(0.0f, 0.0f);
			for (const auto& p : pts)
			{
				approxCenter += p.center;
			}
			approxCenter.x /= static_cast<float>(pts.size());
			approxCenter.y /= static_cast<float>(pts.size());

			// If more than 4, keep the 4 farthest from the approximate center — they are
			// the outermost structural markers.
			if (pts.size() > 4)
			{
				sort(pts.begin(), pts.end(), [&approxCenter](const Marker& a, const Marker& b)
				{
					return norm(a.center - approxCenter) > norm(b.center - approxCenter);
				});
				pts.resize(4);
			}

			// Exact centroid of the 4 selected markers.
			Point2f exactCenter(0.0f, 0.0f);
			for (const auto& p : pts)
			{
				exactCenter += p.center;
			}
			exactCenter.x /= 4.0f;
			exactCenter.y /= 4.0f;

			// BR finder: smallest area/(dist²) ratio — it is physically smaller than the
			// three large finders, so its area is low relative to its corner distance.
			int brIdx = -1;
			float minRatio = 1e9f;
			for (int i = 0; i < 4; ++i)
			{
				const float dist = static_cast<float>(norm(pts[i].center - exactCenter));
				if (dist < 1.0f)
				{
					continue;
				}
				const float ratio = pts[i].area / (dist * dist);
				if (ratio < minRatio)
				{
					minRatio = ratio;
					brIdx = i;
				}
			}
			if (brIdx < 0)
			{
				return false;
			}

			// Sort all 4 by polar angle around exactCenter (counter-clockwise from -π).
			vector<Point2f> sortedPts;
			sortedPts.reserve(4);
			for (const auto& p : pts)
			{
				sortedPts.push_back(p.center);
			}
			sort(sortedPts.begin(), sortedPts.end(), [&exactCenter](const Point2f& a, const Point2f& b)
			{
				return atan2(a.y - exactCenter.y, a.x - exactCenter.x)
				     < atan2(b.y - exactCenter.y, b.x - exactCenter.x);
			});

			// Locate BR in the angle-sorted list.
			const Point2f brPoint = pts[brIdx].center;
			int sortedBrIdx = 0;
			for (int i = 0; i < 4; ++i)
			{
				if (norm(sortedPts[i] - brPoint) < 1.0f)
				{
					sortedBrIdx = i;
					break;
				}
			}

			// Angular order starting from BR: BR → BL → TL → TR (CCW).
			ordered[2] = sortedPts[sortedBrIdx];                  // BR
			ordered[3] = sortedPts[(sortedBrIdx + 1) % 4];        // BL
			ordered[0] = sortedPts[(sortedBrIdx + 2) % 4];        // TL
			ordered[1] = sortedPts[(sortedBrIdx + 3) % 4];        // TR
			return true;
		}
	}

	bool Main(const Mat& srcImg, Mat& disImg)
	{
		if (srcImg.empty())
		{
			return false;
		}

		// Always scale to exactly kMaxDetectionDimension (800px) for detection — matching
		// modify/warp_engine.cpp. This ensures fixed params (block=101, minArea=500, etc.)
		// work correctly for all input sizes, including sub-800px inputs that need upscaling.
		const float scale = kMaxDetectionDimension / static_cast<float>(std::max(srcImg.cols, srcImg.rows));
		Mat smallImg;
		resize(srcImg, smallImg, Size(), scale, scale, scale > 1.0f ? INTER_LINEAR : INTER_AREA);

		vector<Marker> smallMarkers = findMarkerCenters(smallImg);
		vector<Marker> markers;
		markers.reserve(smallMarkers.size());
		for (const auto& pt : smallMarkers)
		{
			markers.push_back({ Point2f(pt.center.x / scale, pt.center.y / scale), pt.area });
		}

		auto useCached = [&]() -> bool
		{
			if (g_lastValidTransform.empty())
			{
				return false;
			}
			Mat warped;
			warpPerspective(srcImg, warped, g_lastValidTransform, Size(kFrameSize, kFrameSize), INTER_NEAREST);
			blockwiseColorMaxAdaptiveThreshold(warped, disImg);
			return true;
		};

		if (markers.size() < 4)
		{
			return useCached();
		}

		array<Point2f, 4> srcPoints{};
		if (!sortFourMarkers(markers, srcPoints))
		{
			return useCached();
		}

		const array<Point2f, 4> dstPoints =
		{ {
			Point2f(kRMin * kFrameSize, kRMin * kFrameSize),
			Point2f(kRMax * kFrameSize, kRMin * kFrameSize),
			Point2f(kRMax * kFrameSize, kRMax * kFrameSize),
			Point2f(kRMin * kFrameSize, kRMax * kFrameSize)
		} };

		const Mat transform = getPerspectiveTransform(srcPoints.data(), dstPoints.data());
		g_lastValidTransform = transform.clone();

		Mat warped;
		warpPerspective(srcImg, warped, transform, Size(kFrameSize, kFrameSize), INTER_NEAREST);
		blockwiseColorMaxAdaptiveThreshold(warped, disImg);
		return true;
	}
} // namespace ImgParse
