#include "pic.h"
#include <vector>
#include <cmath>
#include <algorithm>

namespace ImgParse {

    using namespace std;
    using namespace cv;

    //中文
    //静态全局缓存：保存上一次成功解析的透视变换矩阵
    //
    static Mat lastValidTransform;

    struct Marker {
        Point2f center;
        double area;
    };

    //中文
    //V15 最稳健的三层嵌套轮廓寻找器
    //
    int findLargestChild(int parentIdx, const vector<vector<Point>>& contours, const vector<Vec4i>& hierarchy) {
        int max_idx = -1;
        double max_area = -1.0;
        int child = hierarchy[parentIdx][2];
        while (child >= 0) {
            double area = contourArea(contours[child]);
            if (area > max_area) {
                max_area = area;
                max_idx = child;
            }
            child = hierarchy[child][0];
        }
        return max_idx;
    }

    //中文
    //warp_engine颜色校正：HSV饱和度掩膜 + 自适应阈值二值化（参照modify/warp_engine.cpp）
    //
    void warpEngineColorCorrect(const Mat& imgWarped, Mat& disImg) {
        //中文
        //转灰度，并用HSV饱和度掩膜把彩色区域置白（与warp_engine.cpp保持一致）
        //
        Mat grayWarped;
        cvtColor(imgWarped, grayWarped, COLOR_BGR2GRAY);
        Mat hsvWarped;
        cvtColor(imgWarped, hsvWarped, COLOR_BGR2HSV);
        vector<Mat> hsvCh;
        split(hsvWarped, hsvCh);
        Mat satMask;
        threshold(hsvCh[1], satMask, 180, 255, THRESH_BINARY);
        grayWarped.setTo(255, satMask);

        //中文
        //高斯模糊去噪（参数与warp_engine.cpp缩放逻辑一致，对266px图像取5）
        //
        Mat blurred;
        GaussianBlur(grayWarped, blurred, Size(5, 5), 0);

        //中文
        //高斯加权自适应阈值，block=31对应266px下warp_engine的缩放比例
        //
        Mat binary;
        adaptiveThreshold(blurred, binary, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 31, 10);

        cvtColor(binary, disImg, COLOR_GRAY2BGR);
    }

    //中文
    //二维码主校正流程（定位灰度，warp彩色，二值分块）
    //
    bool processV15(const Mat& srcImg, Mat& /*gray*/, Mat& disImg, bool useHSV) {
        Mat gray;
        if (srcImg.channels() == 3) {
            cvtColor(srcImg, gray, COLOR_BGR2GRAY);
        }
        else {
            gray = srcImg.clone();
        }
        if (useHSV && srcImg.channels() == 3) {
            Mat hsv, binaryMask;
            cvtColor(srcImg, hsv, COLOR_BGR2HSV);
            vector<Mat> hsv_ch;
            split(hsv, hsv_ch);
            threshold(hsv_ch[1], binaryMask, 180, 255, THRESH_BINARY);
            gray.setTo(255, binaryMask);
        }

        int max_dim = std::max(srcImg.cols, srcImg.rows);
        int blur_size = std::max(5, (int)(max_dim * 7.0 / 1920.0));
        if (blur_size % 2 == 0) blur_size++;
        Mat blurred;
        GaussianBlur(gray, blurred, Size(blur_size, blur_size), 0);

        int block_size = std::max(31, (int)(max_dim * 31.0 / 1920.0));
        if (block_size % 2 == 0) block_size++;

        Mat binaryForContours;
        adaptiveThreshold(blurred, binaryForContours, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY_INV, block_size, 10);

        Mat kernel = getStructuringElement(MORPH_CROSS, Size(2, 2));
        Mat closedBinary;
        morphologyEx(binaryForContours, closedBinary, MORPH_CLOSE, kernel);

        vector<vector<Point>> contours;
        vector<Vec4i> hierarchy;
        findContours(closedBinary, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);

        double min_area = std::max(15.0, (double)max_dim * max_dim * 0.000004);
        vector<Marker> markers;

        for (size_t i = 0; i < contours.size(); ++i) {
            int c1 = findLargestChild(i, contours, hierarchy);
            if (c1 < 0) continue;
            int c2 = findLargestChild(c1, contours, hierarchy);
            if (c2 < 0) continue;
            double area0 = contourArea(contours[i]);
            double area1 = contourArea(contours[c1]);
            double area2 = contourArea(contours[c2]);
            if (area0 < min_area) continue;
            double r01 = area0 / max(area1, 1.0);
            double r12 = area1 / max(area2, 1.0);
            if (r01 > 1.2 && r01 < 8.0 && r12 > 1.2 && r12 < 8.0) {
                //中文
                //使用最稳健的外围轮廓计算重心
                //
                Moments M = moments(contours[i]);
                if (M.m00 != 0) {
                    markers.push_back({ Point2f(M.m10 / M.m00, M.m01 / M.m00), area0 });
                }
            }
        }

        double merge_dist = std::max(15.0, (double)max_dim * 0.0078);
        vector<Marker> uniqueMarkers;
        for (const auto& m : markers) {
            bool duplicate = false;
            for (auto& um : uniqueMarkers) {
                if (norm(m.center - um.center) < merge_dist) {
                    if (m.area > um.area) {
                        um.area = m.area;
                        um.center = m.center;
                    }
                    duplicate = true;
                    break;
                }
            }
            if (!duplicate) uniqueMarkers.push_back(m);
        }
        markers = uniqueMarkers;

        if (markers.size() < 3) return false;

        std::sort(markers.begin(), markers.end(), [](const Marker& a, const Marker& b) {
            return a.area > b.area;
            });

        double maxDist = 0;
        int rightAngleIdx = -1;
        for (int i = 0; i < 3; ++i) {
            for (int j = i + 1; j < 3; ++j) {
                double d = norm(markers[i].center - markers[j].center);
                if (d > maxDist) {
                    maxDist = d;
                    rightAngleIdx = 3 - i - j;
                }
            }
        }

        Point2f TL = markers[rightAngleIdx].center;
        Point2f pt1 = markers[(rightAngleIdx + 1) % 3].center;
        Point2f pt2 = markers[(rightAngleIdx + 2) % 3].center;

        Point2f v1 = pt1 - TL;
        Point2f v2 = pt2 - TL;
        double len1 = norm(v1);
        double len2 = norm(v2);

        double legRatio = len1 / max(len2, 1.0);
        if (legRatio < 0.4 || legRatio > 2.5) return false;
        double cosTheta = (v1.x * v2.x + v1.y * v2.y) / max(len1 * len2, 1.0);
        if (std::abs(cosTheta) > 0.75) return false;
        double cross = v1.x * v2.y - v1.y * v2.x;
        Point2f TR, BL;
        if (cross > 0) { TR = pt1; BL = pt2; }
        else { TR = pt2; BL = pt1; }
        Point2f BR;
        bool foundBR = false;
        Point2f expectedBR = TR + BL - TL;
        if (markers.size() > 3) {
            double minDist = 1e9;
            int bestIdx = -1;
            for (size_t i = 3; i < markers.size(); ++i) {
                double d = norm(markers[i].center - expectedBR);
                if (d < minDist) {
                    minDist = d;
                    bestIdx = i;
                }
            }
            if (minDist < max(len1, len2) * 0.4) {
                BR = markers[bestIdx].center;
                foundBR = true;
            }
        }
        if (!foundBR) BR = expectedBR;
        vector<Point2f> srcPoints = { TL, TR, BR, BL };
        vector<Point2f> dstPoints266 = {
            Point2f(21.0f, 21.0f),
            Point2f(245.0f, 21.0f),
            Point2f(245.0f, 245.0f),
            Point2f(21.0f, 245.0f)
        };
        Mat transformMatrix266 = getPerspectiveTransform(srcPoints, dstPoints266);
        lastValidTransform = transformMatrix266.clone();
        //中文
        // 对原图srcImg（BGR彩色）做warpPerspective！
        //
        Mat imgWarped;
        warpPerspective(srcImg, imgWarped, transformMatrix266, Size(266, 266), INTER_LINEAR);

        //中文
        //使用warp_engine颜色校正进行二值化
        //
        warpEngineColorCorrect(imgWarped, disImg);
        return true;
    }

    bool Main(const cv::Mat& srcImg, cv::Mat& disImg) {
        if (srcImg.empty()) return false;
        static int last_cols = 0;
        static int last_rows = 0;
        static int v5_frame_count = 0;
        if (srcImg.cols != last_cols || srcImg.rows != last_rows) {
            last_cols = srcImg.cols;
            last_rows = srcImg.rows;
            v5_frame_count = 0;
            lastValidTransform = Mat();
        }
        double aspect = (double)srcImg.cols / srcImg.rows;
        if (aspect > 0.95 && aspect < 1.05 && srcImg.cols > 200) {
            Mat imgGray;
            if (srcImg.channels() == 3) cvtColor(srcImg, imgGray, COLOR_BGR2GRAY);
            else imgGray = srcImg.clone();
            Mat binRaw;
            threshold(imgGray, binRaw, 0, 255, THRESH_BINARY | THRESH_OTSU);
            disImg.create(266, 266, CV_8UC3);
            float stepX = (float)srcImg.cols / 266.0f;
            float stepY = (float)srcImg.rows / 266.0f;
            for (int r = 0; r < 266; ++r) {
                for (int c = 0; c < 266; ++c) {
                    int px_x = std::min(static_cast<int>((c + 0.5f) * stepX), srcImg.cols - 1);
                    int py_y = std::min(static_cast<int>((r + 0.5f) * stepY), srcImg.rows - 1);
                    uint8_t val = binRaw.at<uint8_t>(py_y, px_x);
                    disImg.at<Vec3b>(r, c) = val ? Vec3b(255, 255, 255) : Vec3b(0, 0, 0);
                }
            }
            return true;
        }
        if (v5_frame_count < 3) {
            v5_frame_count++;
            return false;
        }
        Mat grayNormal;
        if (srcImg.channels() == 3)
            cvtColor(srcImg, grayNormal, COLOR_BGR2GRAY);
        else
            grayNormal = srcImg.clone();
        if (processV15(srcImg, grayNormal, disImg, false)) {
            return true;
        }
        if (srcImg.channels() == 3) {
            Mat grayHSV = grayNormal.clone();
            if (processV15(srcImg, grayHSV, disImg, true)) {
                return true;
            }
        }
        if (!lastValidTransform.empty()) {
            Mat imgWarped;
            //中文
            //兜底：对原彩色图做仿射，warp_engine颜色校正处理
            //
            warpPerspective(srcImg, imgWarped, lastValidTransform, Size(266, 266), INTER_LINEAR);
            warpEngineColorCorrect(imgWarped, disImg);
            return true;
        }
        return false;
    }

} // namespace ImgParse
