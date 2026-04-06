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
		constexpr float kFinderCenter = 21.0f;                 // center of each large 42x42 finder marker (index 12-15 ring: col/row 6..36, centroid=21)
		constexpr float kOppositeFinderCenter = 245.0f;        // 266 - 21: center of TR/BL large markers
		constexpr float kSmallFinderCenter = 252.5f;           // center of the small BR marker in logical frame: (SmallQrPointStart+SmallQrPointEnd)/2 = (246+259)/2
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

		bool sortPointsFromThreeMarkers(const vector<Marker>& markers, array<Point2f, 4>& ordered)
		{
			if (markers.size() < 3)
			{
				return false;
			}

			vector<Marker> m = markers;
			sort(m.begin(), m.end(), [](const Marker& a, const Marker& b)
			{
				return a.area > b.area;
			});
			m.resize(3);

			double maxDist = 0.0;
			int rightAngleIdx = -1;
			for (int i = 0; i < 3; ++i)
			{
				for (int j = i + 1; j < 3; ++j)
				{
					const double d = norm(m[i].center - m[j].center);
					if (d > maxDist)
					{
						maxDist = d;
						rightAngleIdx = 3 - i - j;
					}
				}
			}
			if (rightAngleIdx < 0)
			{
				return false;
			}

			const Point2f tl = m[rightAngleIdx].center;
			const Point2f p1 = m[(rightAngleIdx + 1) % 3].center;
			const Point2f p2 = m[(rightAngleIdx + 2) % 3].center;
			const Point2f v1 = p1 - tl;
			const Point2f v2 = p2 - tl;
			const double len1 = norm(v1);
			const double len2 = norm(v2);
			const double legRatio = len1 / std::max(len2, 1.0);
			if (legRatio < 0.4 || legRatio > 2.5)
			{
				return false;
			}
			const double cosTheta = (v1.x * v2.x + v1.y * v2.y) / std::max(len1 * len2, 1.0);
			if (std::abs(cosTheta) > 0.75)
			{
				return false;
			}

			Point2f tr, bl;
			const float cross = v1.x * v2.y - v1.y * v2.x;
			if (cross > 0)
			{
				tr = p1;
				bl = p2;
			}
			else
			{
				tr = p2;
				bl = p1;
			}
			const Point2f br = tr + bl - tl;
			ordered[0] = tl;
			ordered[1] = tr;
			ordered[2] = br;
			ordered[3] = bl;
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
			warpPerspective(srcImg, warped, g_lastValidTransform, Size(kFrameSize, kFrameSize), INTER_LINEAR);
			blockwiseColorMaxAdaptiveThreshold(warped, disImg);
			return true;
		};

		if (markers.size() < 3)
		{
			return useCached();
		}

		// Sort by area descending. The 3 largest are the big 42x42 finder markers (TL, TR, BL).
		// This mirrors the V15process approach: always anchor on the three large structural markers,
		// ignoring smaller candidates that can flicker frame-to-frame and cause tearing.
		sort(markers.begin(), markers.end(), [](const Marker& a, const Marker& b)
		{
			return a.area > b.area;
		});

		const vector<Marker> largeMarkers(markers.begin(), markers.begin() + 3);
		array<Point2f, 4> srcPoints{};
		if (!sortPointsFromThreeMarkers(largeMarkers, srcPoints))
		{
			return useCached();
		}

		// srcPoints: [0]=TL, [1]=TR, [2]=estimated BR (parallelogram TR+BL-TL), [3]=BL.
		// Optionally refine the BR point: look for a small marker (area significantly below the
		// large ones) that lies within 35% of the TL-TR side length from the estimated BR position.
		// 35% covers the geometric offset between the parallelogram estimate and the actual small
		// marker position, plus typical detection noise, without accepting distant false positives.
		const float searchRadius = static_cast<float>(norm(srcPoints[1] - srcPoints[0])) * 0.35f;
		// Reject any candidate whose area is more than half that of the smallest large marker;
		// the small BR finder pattern (14x14 logical) is at most ~1/9 the area of a large one (42x42),
		// so 0.5 comfortably separates small from large while tolerating detection variation.
		const float maxSmallArea = largeMarkers[2].area * 0.5f;
		bool foundSmallBR = false;
		for (size_t i = 3; i < markers.size(); ++i)
		{
			if (markers[i].area > maxSmallArea)
			{
				continue;
			}
			const float dist = static_cast<float>(norm(markers[i].center - srcPoints[2]));
			if (dist < searchRadius)
			{
				srcPoints[2] = markers[i].center;
				foundSmallBR = true;
				break;
			}
		}

		const float brDstCoord = foundSmallBR ? kSmallFinderCenter : kOppositeFinderCenter;
		const array<Point2f, 4> dstPoints =
		{ {
			Point2f(kFinderCenter, kFinderCenter),
			Point2f(kOppositeFinderCenter, kFinderCenter),
			Point2f(brDstCoord, brDstCoord),
			Point2f(kFinderCenter, kOppositeFinderCenter)
		} };

		const Mat transform = getPerspectiveTransform(srcPoints.data(), dstPoints.data());
		g_lastValidTransform = transform.clone();

		Mat warped;
		warpPerspective(srcImg, warped, transform, Size(kFrameSize, kFrameSize), INTER_LINEAR);
		blockwiseColorMaxAdaptiveThreshold(warped, disImg);
		return true;
	}
} // namespace ImgParse
