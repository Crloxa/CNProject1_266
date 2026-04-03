#include "pic.h"

#include <algorithm>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <vector>

namespace ImgParse
{
	using namespace cv;
	using namespace std;

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

	vector<Point3f> FindMarkerCenters(const Mat& input, int ch)
	{
		Mat gray;
		if (ch == 3)
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
		int blockSize = std::max(31, static_cast<int>(maxDim * 101.0 / 800.0));
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

		const float areaThreshold = std::max(120.0f, 500.0f * (maxDim / 800.0f) * (maxDim / 800.0f));
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

		const float mergeDist = std::max(24.0f, 100.0f * (maxDim / 800.0f));
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

		// 统一先缩放到约 800 像素工作尺度：降低噪声与计算量，同时稳定轮廓参数范围。
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

		if (centers.size() < 4)
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
			// 选取距离整体中心更远的 4 点，优先保留外层定位块，抑制中心附近伪候选点。
			sort(centers.begin(), centers.end(), [&approxCenter](Point3f a, Point3f b)
			{
				return norm(Point2f(a.x, a.y) - approxCenter) > norm(Point2f(b.x, b.y) - approxCenter);
			});
			centers.resize(4);
		}

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

		const Point2f br = sortedPts[sortedBrIdx];
		const Point2f bl = sortedPts[(sortedBrIdx + 1) % 4];
		const Point2f tl = sortedPts[(sortedBrIdx + 2) % 4];
		const Point2f tr = sortedPts[(sortedBrIdx + 3) % 4];

		const float padX = outWidth * 0.05225f;
		const float padY = outHeight * 0.05225f;
		const float correct = outWidth * 0.0160f;

		const vector<Point2f> src = { tl, tr, br, bl };
		const vector<Point2f> dst = {
			Point2f(padX, padY),
			Point2f(outWidth - 1 - padX, padY),
			Point2f(outWidth - 1 - padX - correct, outHeight - 1 - padY - correct),
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

		Mat warpedGray;
		cvtColor(warped, warpedGray, COLOR_BGR2GRAY);
		Mat warpedBin;
		threshold(warpedGray, warpedBin, 0, 255, THRESH_BINARY | THRESH_OTSU);
		cvtColor(warpedBin, disImg, COLOR_GRAY2BGR);
		return true;
	}
}
