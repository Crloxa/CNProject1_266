#include "pic.h"

#include <algorithm>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <vector>

namespace ImgParse
{
	using namespace cv;
	using namespace std;

	constexpr float kWarpPaddingRatio = 0.05225f;
	constexpr float kBrCornerCompensationRatio = 0.0160f;

	struct Marker
	{
		Point2f center;
		double area;
	};

	int findLargestChild(int parentIdx, const vector<vector<Point>>& contours, const vector<Vec4i>& hierarchy)
	{
		int maxIdx = -1;
		double maxArea = -1.0;
		int child = hierarchy[parentIdx][2];
		while (child >= 0)
		{
			const double area = contourArea(contours[child]);
			if (area > maxArea)
			{
				maxArea = area;
				maxIdx = child;
			}
			child = hierarchy[child][0];
		}
		return maxIdx;
	}

	void blockwiseColorMaxAdaptiveThreshold(const Mat& imgColor, Mat& binImg, int blockSize = 19, int bias = 10)
	{
		const int h = imgColor.rows;
		const int w = imgColor.cols;
		const int nBlockY = (h + blockSize - 1) / blockSize;
		const int nBlockX = (w + blockSize - 1) / blockSize;
		vector<vector<int>> thresholds(nBlockY, vector<int>(nBlockX, 128));

		for (int by = 0; by < nBlockY; ++by)
		{
			for (int bx = 0; bx < nBlockX; ++bx)
			{
				vector<int> samples;
				const int y0 = by * blockSize;
				const int y1 = std::min(y0 + blockSize, h);
				const int x0 = bx * blockSize;
				const int x1 = std::min(x0 + blockSize, w);
				samples.reserve((y1 - y0) * (x1 - x0));
				for (int y = y0; y < y1; ++y)
				{
					for (int x = x0; x < x1; ++x)
					{
						const Vec3b pix = imgColor.at<Vec3b>(y, x);
						samples.push_back(std::max(pix[0], std::max(pix[1], pix[2])));
					}
				}
				if (samples.empty())
				{
					continue;
				}
				std::sort(samples.begin(), samples.end());
				const int n = static_cast<int>(samples.size());
				const int lowIdx = n / 10;
				const int highIdx = n - n / 10 - 1;
				const int blackMax = samples[lowIdx];
				const int whiteMin = samples[highIdx];
				int t = (blackMax + whiteMin) / 2 + bias;
				t = std::max(0, std::min(255, t));
				thresholds[by][bx] = t;
			}
		}

		binImg.create(h, w, CV_8UC3);
		for (int y = 0; y < h; ++y)
		{
			const int by = y / blockSize;
			for (int x = 0; x < w; ++x)
			{
				const int bx = x / blockSize;
				const int t = thresholds[by][bx];
				const Vec3b pix = imgColor.at<Vec3b>(y, x);
				const int mx = std::max(pix[0], std::max(pix[1], pix[2]));
				binImg.at<Vec3b>(y, x) = (mx > t) ? Vec3b(255, 255, 255) : Vec3b(0, 0, 0);
			}
		}
	}

	bool extractThreeMarkerCorners(const vector<Point3f>& centers, Point2f& tl, Point2f& tr, Point2f& bl)
	{
		if (centers.size() != 3)
		{
			return false;
		}
		double maxDist = 0.0;
		int rightAngleIdx = -1;
		for (int i = 0; i < 3; ++i)
		{
			for (int j = i + 1; j < 3; ++j)
			{
				const double d = norm(Point2f(centers[i].x, centers[i].y) - Point2f(centers[j].x, centers[j].y));
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

		tl = Point2f(centers[rightAngleIdx].x, centers[rightAngleIdx].y);
		const Point2f p1 = Point2f(centers[(rightAngleIdx + 1) % 3].x, centers[(rightAngleIdx + 1) % 3].y);
		const Point2f p2 = Point2f(centers[(rightAngleIdx + 2) % 3].x, centers[(rightAngleIdx + 2) % 3].y);
		const Point2f v1 = p1 - tl;
		const Point2f v2 = p2 - tl;
		const double len1 = norm(v1);
		const double len2 = norm(v2);
		if (len1 <= 1e-6 || len2 <= 1e-6)
		{
			return false;
		}

		const double legRatio = len1 / len2;
		if (legRatio < 0.4 || legRatio > 2.5)
		{
			return false;
		}
		const double cosTheta = (v1.x * v2.x + v1.y * v2.y) / (len1 * len2);
		if (std::abs(cosTheta) > 0.75)
		{
			return false;
		}

		const double cross = v1.x * v2.y - v1.y * v2.x;
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
		return true;
	}

	vector<Point3f> FindMarkerCenters(const Mat& input, int channels)
	{
		Mat gray;
		if (channels == 3)
		{
			Mat hsv;
			Mat binaryMask;
			cvtColor(input, hsv, COLOR_BGR2HSV);
			cvtColor(input, gray, COLOR_BGR2GRAY);

			vector<Mat> hsvCh;
			split(hsv, hsvCh);
			threshold(hsvCh[1], binaryMask, 180, 255, THRESH_BINARY);
			gray.setTo(255, binaryMask);
		}
		else
		{
			gray = input.clone();
		}

		const int maxDim = std::max(input.cols, input.rows);
		int blockSize = std::max(31, static_cast<int>(maxDim * 31.0 / 1920.0));
		if ((blockSize & 1) == 0)
		{
			++blockSize;
		}
		Mat binary;
		adaptiveThreshold(gray, binary, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY_INV, blockSize, 15);
		Mat kernel = getStructuringElement(MORPH_CROSS, Size(2, 2));
		Mat closedBinary;
		morphologyEx(binary, closedBinary, MORPH_CLOSE, kernel);

		vector<vector<Point>> contours;
		vector<Vec4i> hierarchy;
		findContours(closedBinary, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);

		const float areaThreshold = static_cast<float>(std::max(15.0, static_cast<double>(maxDim) * maxDim * 0.000004));
		vector<Point3f> centers;
		for (size_t i = 0; i < contours.size(); ++i)
		{
			int kidIdx = hierarchy[i][2];
			int cnt = 0;
			while (kidIdx != -1)
			{
				++cnt;
				kidIdx = hierarchy[kidIdx][2];
				if (cnt >= 2)
				{
					break;
				}
			}
			if (cnt >= 2)
			{
				Moments mu = moments(contours[i], false);
				if (mu.m00 >= areaThreshold)
				{
					centers.push_back(Point3f(static_cast<float>(mu.m10 / mu.m00), static_cast<float>(mu.m01 / mu.m00), static_cast<float>(mu.m00)));
				}
			}
		}

		const float mergeDist = static_cast<float>(std::max(15.0, maxDim * 0.0078));
		vector<Point3f> merged;
		for (const auto& pt : centers)
		{
			bool isNew = true;
			for (auto& m : merged)
			{
				if (norm(Point2f(pt.x, pt.y) - Point2f(m.x, m.y)) < mergeDist)
				{
					isNew = false;
					if (pt.z > m.z)
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

	bool Main(const Mat& srcImg, Mat& disImg)
	{
		if (srcImg.empty())
		{
			return false;
		}

		const int outWidth = 266;
		const int outHeight = 266;
		const int channels = srcImg.channels();
		if (channels != 1 && channels != 3)
		{
			return false;
		}

		// Normalize to ~800px working scale to reduce noise and keep contour parameters stable.
		const float scale = 800.0f / static_cast<float>(std::max(srcImg.cols, srcImg.rows));
		Mat smallImg;
		resize(srcImg, smallImg, Size(), scale, scale, INTER_AREA);

		vector<Point3f> smallCenters = FindMarkerCenters(smallImg, channels);
		vector<Point3f> centers;
		centers.reserve(smallCenters.size());
		for (auto pt : smallCenters)
		{
			centers.push_back(Point3f(pt.x / scale, pt.y / scale, pt.z));
		}

		if (centers.size() < 3)
		{
			return false;
		}

		Point2f approxCenter(0, 0);
		for (const auto& p : centers)
		{
			approxCenter += Point2f(p.x, p.y);
		}
		approxCenter.x /= static_cast<float>(centers.size());
		approxCenter.y /= static_cast<float>(centers.size());

		if (centers.size() > 4)
		{
			// Keep the 4 points farther from global center to suppress central false candidates.
			sort(centers.begin(), centers.end(), [&approxCenter](Point3f a, Point3f b)
			{
				return norm(Point2f(a.x, a.y) - approxCenter) > norm(Point2f(b.x, b.y) - approxCenter);
			});
			centers.resize(4);
		}

		Point2f tl, tr, br, bl;
		if (centers.size() == 3)
		{
			if (!extractThreeMarkerCorners(centers, tl, tr, bl))
			{
				return false;
			}
			br = tr + bl - tl;
		}
		else
		{
			Point2f exactCenter(0, 0);
			for (const auto& p : centers)
			{
				exactCenter += Point2f(p.x, p.y);
			}
			exactCenter.x /= 4.0f;
			exactCenter.y /= 4.0f;

			int brIdx = -1;
			float minRatio = 1e9f;
			for (int i = 0; i < 4; ++i)
			{
				const float dist = norm(Point2f(centers[i].x, centers[i].y) - exactCenter);
				if (dist <= 1e-6f)
				{
					continue;
				}
				const float area = centers[i].z;
				const float ratio = area / (dist * dist);
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

			const Point2f brPoint(centers[brIdx].x, centers[brIdx].y);
			vector<Point2f> sortedPts;
			sortedPts.reserve(4);
			for (const auto& p : centers)
			{
				sortedPts.push_back(Point2f(p.x, p.y));
			}
			sort(sortedPts.begin(), sortedPts.end(), [&exactCenter](Point2f a, Point2f b)
			{
				return atan2(a.y - exactCenter.y, a.x - exactCenter.x) < atan2(b.y - exactCenter.y, b.x - exactCenter.x);
			});

			int sortedBrIdx = 0;
			for (int i = 0; i < 4; ++i)
			{
				if (norm(sortedPts[i] - brPoint) < 1.0f)
				{
					sortedBrIdx = i;
					break;
				}
			}
			br = sortedPts[sortedBrIdx];
			bl = sortedPts[(sortedBrIdx + 1) % 4];
			tl = sortedPts[(sortedBrIdx + 2) % 4];
			tr = sortedPts[(sortedBrIdx + 3) % 4];
		}

		const float padX = outWidth * kWarpPaddingRatio;
		const float padY = outHeight * kWarpPaddingRatio;
		// Keep warp_engine-style BR-only compensation to improve bottom-edge alignment under perspective.
		const float brCornerCompensation = outWidth * kBrCornerCompensationRatio;

		const vector<Point2f> src = { tl, tr, br, bl };
		const vector<Point2f> dst = {
			Point2f(padX, padY),
			Point2f(outWidth - 1 - padX, padY),
			Point2f(outWidth - 1 - padX - brCornerCompensation, outHeight - 1 - padY - brCornerCompensation),
			Point2f(padX, outHeight - 1 - padY)
		};

		Mat srcColor;
		if (channels == 3)
		{
			srcColor = srcImg;
		}
		else
		{
			cvtColor(srcImg, srcColor, COLOR_GRAY2BGR);
		}

		const Mat transform = getPerspectiveTransform(src, dst);
		Mat warped;
		warpPerspective(srcColor, warped, transform, Size(outWidth, outHeight), INTER_LINEAR);

		Mat warpedBin;
		blockwiseColorMaxAdaptiveThreshold(warped, warpedBin, 19, 10);
		disImg = warpedBin;
		return true;
	}
}
