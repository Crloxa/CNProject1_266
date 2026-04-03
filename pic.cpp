#include "pic.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <vector>

namespace ImgParse
{
	using namespace cv;
	using namespace std;

	namespace
	{
		constexpr int kFrameSize = 266;
		constexpr float kFinderCenter = 21.0f;                 // aligned with current encoder finder-center mapping
		constexpr float kOppositeFinderCenter = 245.0f;        // 266 - 21
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
			const int maxDim = std::max(input.cols, input.rows);
			int adaptiveBlock = std::max(31, static_cast<int>(maxDim * 31.0 / 1920.0));
			if ((adaptiveBlock & 1) == 0)
			{
				++adaptiveBlock;
			}
			const double minArea = std::max(15.0, static_cast<double>(maxDim) * maxDim * 0.000004);
			const float mergeDistance = std::max(15.0f, static_cast<float>(maxDim) * 0.0078f);

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

		bool sortPointsFromFourMarkers(const vector<Marker>& markers, array<Point2f, 4>& ordered)
		{
			if (markers.size() < 4)
			{
				return false;
			}

			vector<Marker> candidates = markers;
			Point2f center(0.f, 0.f);
			for (const auto& p : candidates)
			{
				center += p.center;
			}
			center *= (1.f / static_cast<float>(candidates.size()));
			if (candidates.size() > 4)
			{
				sort(candidates.begin(), candidates.end(), [&center](const Marker& a, const Marker& b)
				{
					return norm(a.center - center) > norm(b.center - center);
				});
				candidates.resize(4);
			}

			Point2f exactCenter(0.f, 0.f);
			for (const auto& p : candidates)
			{
				exactCenter += p.center;
			}
			exactCenter *= 0.25f;

			int brIdx = -1;
			float minRatio = std::numeric_limits<float>::max();
			for (int i = 0; i < 4; ++i)
			{
				const float dist = std::max(norm(candidates[i].center - exactCenter), 1.0f);
				const float ratio = candidates[i].area / (dist * dist);
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

			vector<Point2f> sortedPts;
			sortedPts.reserve(4);
			for (const auto& p : candidates)
			{
				sortedPts.push_back(p.center);
			}
			sort(sortedPts.begin(), sortedPts.end(), [&exactCenter](const Point2f& a, const Point2f& b)
			{
				return atan2(a.y - exactCenter.y, a.x - exactCenter.x) < atan2(b.y - exactCenter.y, b.x - exactCenter.x);
			});

			int sortedBrIdx = 0;
			for (int i = 0; i < 4; ++i)
			{
				if (norm(sortedPts[i] - candidates[brIdx].center) < 1.0f)
				{
					sortedBrIdx = i;
					break;
				}
			}

			ordered[2] = sortedPts[sortedBrIdx];
			ordered[3] = sortedPts[(sortedBrIdx + 1) % 4];
			ordered[0] = sortedPts[(sortedBrIdx + 2) % 4];
			ordered[1] = sortedPts[(sortedBrIdx + 3) % 4];
			return true;
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

		const float scale = 800.0f / static_cast<float>(std::max(srcImg.cols, srcImg.rows));
		Mat smallImg;
		resize(srcImg, smallImg, Size(), scale, scale, INTER_AREA);

		vector<Marker> smallMarkers = findMarkerCenters(smallImg);
		vector<Marker> markers;
		markers.reserve(smallMarkers.size());
		for (const auto& pt : smallMarkers)
		{
			markers.push_back({ Point2f(pt.center.x / scale, pt.center.y / scale), pt.area });
		}

		array<Point2f, 4> srcPoints{};
		bool sorted = sortPointsFromFourMarkers(markers, srcPoints);
		if (!sorted)
		{
			sorted = sortPointsFromThreeMarkers(markers, srcPoints);
		}
		if (!sorted)
		{
			if (g_lastValidTransform.empty())
			{
				return false;
			}
			Mat warped;
			warpPerspective(srcImg, warped, g_lastValidTransform, Size(kFrameSize, kFrameSize), INTER_LINEAR);
			blockwiseColorMaxAdaptiveThreshold(warped, disImg);
			return true;
		}

		const array<Point2f, 4> dstPoints =
		{ {
			Point2f(kFinderCenter, kFinderCenter),
			Point2f(kOppositeFinderCenter, kFinderCenter),
			Point2f(kOppositeFinderCenter, kOppositeFinderCenter),
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
