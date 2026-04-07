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
		constexpr float kFinderCenter = 21.0f;                 // TL/TR/BL large marker center in output frame
		constexpr float kOppositeFinderCenter = 245.0f;        // TR/BL large marker center in output frame
		constexpr float kBRFinderCenter = 253.5f;              // BR marker center in output frame
		// Detection parameters tuned at 800px; scaled proportionally for other resolutions.
		constexpr float kRefDimension = 800.0f;
		constexpr int kRefAdaptiveBlock = 101;
		constexpr double kRefMinArea = 500.0;
		constexpr float kRefMergeDistance = 100.0f;
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
			// Scale detection parameters proportionally to image size relative to the
			// reference resolution at which they were tuned (800px).
			const int maxDim = std::max(input.cols, input.rows);
			int adaptiveBlock = std::max(21, static_cast<int>(static_cast<double>(maxDim) * kRefAdaptiveBlock / kRefDimension));
			if ((adaptiveBlock & 1) == 0)
			{
				++adaptiveBlock;
			}
			const double minArea = std::max(50.0, static_cast<double>(maxDim) * maxDim * kRefMinArea / (kRefDimension * kRefDimension));
			const float mergeDistance = std::max(15.0f, static_cast<float>(maxDim) * kRefMergeDistance / kRefDimension);

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
			adaptiveThreshold(gray, binary, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY_INV, adaptiveBlock, 15);
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
					if (mu.m00 >= minArea)
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
					if (norm(pt.center - m.center) < mergeDistance)
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

		// Detect directly on the original image; findMarkerCenters scales its own
		// parameters to the actual image size.
		vector<Marker> markers = findMarkerCenters(srcImg);

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

		// Sort by area descending; take the 3 largest (the stable 42x42 large markers).
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
		// dst points match the logical 266x266 frame layout exactly as specified.
		const array<Point2f, 4> dstPoints =
		{ {
			Point2f(kFinderCenter,         kFinderCenter),
			Point2f(kOppositeFinderCenter, kFinderCenter),
			Point2f(kBRFinderCenter,       kBRFinderCenter),
			Point2f(kFinderCenter,         kOppositeFinderCenter)
		} };

		const Mat transform = getPerspectiveTransform(srcPoints.data(), dstPoints.data());
		g_lastValidTransform = transform.clone();

		Mat warped;
		warpPerspective(srcImg, warped, transform, Size(kFrameSize, kFrameSize), INTER_LINEAR);
		blockwiseColorMaxAdaptiveThreshold(warped, disImg);
		return true;
	}
} // namespace ImgParse
