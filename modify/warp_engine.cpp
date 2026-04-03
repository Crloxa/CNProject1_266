#include <iostream>
#include <opencv2/opencv.hpp>//环境自己配opencv_world42x.dll要同文件夹并且不能带d
#include <vector>
#include <cmath>
#include <algorithm>
#include <chrono>

using namespace cv;
using namespace std;
/*==============================================================================
                                    [识别模块]
        input:输入图像             ch:通道数             功能:寻找定位块        
*///============================================================================
vector<Point3f> FindMarkerCenters(const Mat& input, int ch) {                               //python传进来的格式是BGR
    Mat gray;//声明灰度图矩阵
    if (ch == 3) {                                                                          //是否三通道彩色图
        Mat hsv, BinaryMask;                                                                //声明"HSV色彩空间"矩阵与"黑白掩膜"矩阵
        cvtColor(input, hsv, COLOR_BGR2HSV);                                                //把BGR转成HSV色彩空间
        cvtColor(input, gray, COLOR_BGR2GRAY);                                              //把BGR转成单通道灰度图
        
        vector<Mat> hsv_ch;
        split(hsv, hsv_ch);                                                                 //拆分成"色相[0]","饱和度[1]","亮度[2]"
        threshold(hsv_ch[1], BinaryMask, 180, 255, THRESH_BINARY);                          //二值化操作，把hsv_ch[1]里饱和度超过180的区域转成255(纯白),THRESH_BINARY是二值化参数,现在我们把所有白色膜区域，全部在灰度图里全部变成白色，，其他的一律定成黑色，这样子整张图就只剩四角定位块和中间零散黑点        
        gray.setTo(255, BinaryMask);                                                        //把二值图的掩膜打在灰度图上,
    } else {  
        gray = input.clone();                                                               //黑白图直接用了
    }

    /*//========================================================================
    不要问我为啥定位块我不同意做彩色，自己看24行，我使用饱和度来判断，
    以上方案改不了一点，所有的参数和色彩空间已经无法进一步调优，我手搓
    好几种新方案均不如这种，没必要调优这一部分了，已经是极限了
    *///========================================================================

    Mat binary;                                                                             //声明局部二值化矩阵
    adaptiveThreshold(gray,binary,255,ADAPTIVE_THRESH_GAUSSIAN_C,THRESH_BINARY_INV,101,15); //高斯加权处理近似，不要改第二个参数
    Mat kernel = getStructuringElement(MORPH_CROSS, Size(2, 2));                            //定义一个L形手术刀，在后面做单像素摩尔纹处理
    Mat closed_binary;                                                                      //声明闭运算处理后的黑白图像
    morphologyEx(binary, closed_binary, MORPH_CLOSE, kernel);                               //开始做闭运算处理

    /*//========================================================================
    上面的模块就别删了，理论上是矩形开运算，因为我搞反了，因为我需要吃掉白色摩尔纹
    所以应该开运算，但是误打误撞，闭运算L型手术刀效果很好，各位活爹，上面改的代码和参数
    调了两节课，不敢动，能跑就行，各位活爹也都别动了
    *///========================================================================

    // =========================================================================
    //[调试窗口1]展示形态学缝合后的黑白反转二值图
    // static bool window_init = false;
    // if (!window_init) {
    //     namedWindow("Debug: Binarized", WINDOW_NORMAL);
    //     window_init = true;
    // }
    // imshow("Debug: Binarized", closed_binary);
    // =========================================================================

    vector<vector<Point>> contours;                                                         //储存轮廊点集合
    vector<Vec4i> hierarchy;                                                                //储存轮廊父子关系
    findContours(closed_binary, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);       //opencv函数，输入图像，轮廊点容器，轮廊关系容器，拓扑结构，轮廓近似方法
    
    /*//========================================================================
        mod
        RETR_EXTERNAL	只找最外层。
        RETR_LIST	    找所有轮廓，但不建立父子关系。
        RETR_CCOMP	    建立两层级。顶层是外框，第二层是里面的孔。	
        RETR_TREE	    建立完整树形结构，实际那个定位块会找到多维嵌套
        compu参数也换不了一点，上面模块也不要动了
    *///========================================================================

    vector<Point3f> centers;                                                                //存符合条件的中心点和它的面积
    for (size_t i=0;i<contours.size();i++) {
        int kid_idx=hierarchy[i][2];                                                        //记录i个定位块的第一个子轮廊
        int cnt = 0;                                                                        //层数
        while(kid_idx!=-1) {                                                                //类似跳表，一路进去数几层，其实2层就能跳出来了
            cnt++;
            kid_idx=hierarchy[kid_idx][2];
            if(cnt>=2)break;                                                                //优化性能
        }
        if (cnt >= 2) {                                                                     //两层了，杀掉无嵌套噪点
            Moments mu = moments(contours[i], false);                                       //提取特征矩
            if (mu.m00 >= 500)                                                               //m00是面积矩，用来杀掉小白点干扰的
                centers.push_back(Point3f(mu.m10 / mu.m00, mu.m01 / mu.m00, mu.m00));       //计算质心，同时存下面积
        }
    }
    //不要再问我为什么定位块不做成彩色，然后定位块缩小这种唐人方案，自己看第76行
    vector<Point3f> merged;                                                                 //存最后的定位块
    for (const auto& pt : centers) {                                                        //合并去重。防止距离过进产生多个定位点干扰
        bool is_new = true;
        for (auto& m : merged) {
            if (norm(Point2f(pt.x, pt.y) - Point2f(m.x, m.y)) < 100.0) {                    //杀掉距离过近的定位快
                is_new = false; 
                if (pt.z > m.z) m = pt;                                                     //通过面积判断主要中心定位块
                break; 
            }
        }
        if (is_new) merged.push_back(pt);
    }
    return merged;                                                                          //返回我们找到的所有定位块
}

/*==============================================================================
                                    [接口函数]
*///============================================================================
extern "C" {
auto start_time = std::chrono::high_resolution_clock::now();                                //性能计时器，觉得烦可以删掉
__declspec(dllexport) bool ExtractQRCode( 
    unsigned char* in_data, int width, int height, int channels,
    unsigned char* out_data, int out_width, int out_height) 
{
    Mat img(height, width, (channels == 3) ? CV_8UC3 : CV_8UC1, in_data);                   //把py传进来的指针变成opencv能处理的mat对象，这里0拷贝，速度会快一点
    Mat out_img(out_height, out_width, CV_8UC3, out_data);                                  //彩色图
    float scale = 800.0f / max(width, height);                                              //缩放比例，杀摩尔纹，以及提高计算速度，4k图要找到什么时候，我只能搜小处理,然后超不多一张图片的时间消耗差不多是50ms
    Mat small_img;                                                                          //小图像                                                                                  
    resize(img, small_img, Size(), scale, scale, INTER_AREA);                               //ai协助的INTER_AREA：区域插值法，这是缩小图像时最能抗锯齿，抗摩尔纹的插值方式。
    vector<Point3f> small_centers = FindMarkerCenters(small_img, channels);                 //然后我们用识别模块，开始找
    vector<Point3f> centers;
    for (auto pt : small_centers)
        centers.push_back(Point3f(pt.x / scale, pt.y / scale, pt.z));

    // =========================================================================
    // [调试窗口2]定位块标记，取消注释即可观看标记
    Scalar drawColor(0, 255, 0);
    for (size_t i = 0; i < centers.size(); i++) {
        Point2f draw_pt(centers[i].x, centers[i].y);
        circle(img, draw_pt, 12, drawColor, 2);
        string label = to_string(i);
        putText(img, label, Point(draw_pt.x + 10, draw_pt.y + 10), 
                FONT_HERSHEY_SIMPLEX, 1.0, drawColor, 2);
    }
    // =========================================================================
    
    Point2f tl, tr, br, bl;                                                                 //四角变量
    if (centers.size() >= 4) {                                            
        Point2f approx_center(0, 0);                                                        //近似中心误差排除法
        for(auto p : centers) approx_center += Point2f(p.x, p.y);
        approx_center.x /= centers.size(); approx_center.y /= centers.size();               //计算近似中心
        if (centers.size() > 4) {
            sort(centers.begin(), centers.end(), [&approx_center](Point3f a, Point3f b) {   //将点按照距离近似中心的远近降序排列
                return norm(Point2f(a.x, a.y) - approx_center) > norm(Point2f(b.x, b.y) - approx_center);
            });
            centers.resize(4); 
        }
        Point2f exact_center(0, 0);
        for(auto p : centers) exact_center += Point2f(p.x, p.y);
        exact_center.x /= 4.0f; exact_center.y /= 4.0f;                                     //计算精确中心

        // =====================================================================
        //                         [抗透视反转模块]
        int br_idx = -1;
        float min_ratio = 1e9;
        for (int i = 0; i < 4; i++) {
            float dist = norm(Point2f(centers[i].x, centers[i].y) - exact_center);
            float area = centers[i].z; 
            float ratio = area / (dist * dist); 
            if (ratio < min_ratio) {
                min_ratio = ratio;
                br_idx = i; 
            }
        }
        Point2f br_point = Point2f(centers[br_idx].x, centers[br_idx].y);
        vector<Point2f> sorted_pts;
        for(auto p : centers) sorted_pts.push_back(Point2f(p.x, p.y));
        sort(sorted_pts.begin(), sorted_pts.end(), [&exact_center](Point2f a, Point2f b) {
            return atan2(a.y - exact_center.y, a.x - exact_center.x) < atan2(b.y - exact_center.y, b.x - exact_center.x);
        });
        int sorted_br_idx = 0;
        for(int i = 0; i < 4; i++) {
            if(norm(sorted_pts[i] - br_point) < 1.0f) { 
                sorted_br_idx = i;
                break;
            }
        }

    //==========================================================================
    //                            [旋转模块]
        br = sorted_pts[sorted_br_idx];
        bl = sorted_pts[(sorted_br_idx + 1) % 4];                                           // BR 的下一个是 BL
        tl = sorted_pts[(sorted_br_idx + 2) % 4];                                           // BR 的对角是 TL
        tr = sorted_pts[(sorted_br_idx + 3) % 4];       
                                            // BR 的上一个是 TR
    //==========================================================================
    //                          [白边消除模块]
        float pad_x = out_width * 0.05225; 
        float pad_y = out_height * 0.05225;

    //==========================================================================
    //                    [方向定位块拉伸变形修正模块]
        float correct = out_width * 0.0160f;                                                //我人工调的参数不要动了  
        vector<Point2f> src = { tl, tr, br, bl };
        vector<Point2f> dst = { 
            Point2f(pad_x, pad_y), 
            Point2f(out_width - 1 - pad_x, pad_y), 
            Point2f(out_width - 1 - pad_x - correct, out_height - 1 - pad_y - correct), 
            Point2f(pad_x, out_height - 1 - pad_y) 
        };
        Mat M = getPerspectiveTransform(src, dst);
        warpPerspective(img, out_img, M, Size(out_width, out_height), INTER_NEAREST);
    } 

    //==========================================================================
    //[调试窗口3]                   [性能计算模块]
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    auto duration_us = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
    std::cout << ">>> [Warp Engine] 处理耗时: " << duration_ms << " ms (" << duration_us << " us)" << std::endl;

    //==========================================================================
    return true;
}
}