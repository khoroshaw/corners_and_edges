#include<iostream>
#include<cmath>
#include<opencv2/core.hpp>
#include<opencv2/imgproc.hpp>
#include<opencv2/imgcodecs.hpp>
#include<opencv2/highgui.hpp>
#include<vector>
#include<queue>
#include<algorithm>

void calc_sobel(int height, int width, cv::Mat& DMap, cv::Mat& GMap)
{
    cv::Mat SobelTempMap(3, width, CV_64F);
    for(int i = 0; i < height; i++)
    {
        double * GaussPtr = GMap.ptr<double>(i);
        int idx_temp_row = i % 3;
        double * SobelTempPtr = SobelTempMap.ptr<double>(idx_temp_row);
        for(int j = 0; j < width; j++)
        {
            SobelTempPtr[j] = 0;
            for(int k = -1; k <= 1; k = k + 2)
            {
                int idx_gauss = std::min(width - 1, std::max(0, j + k));
                SobelTempPtr[j] += k * GaussPtr[idx_gauss];
            }
        }
        if(i > 0)
        {
            int idx_calc = i - 1;
            double * DPtr = DMap.ptr<double>(idx_calc);
            for(int k = -1; k <= 1; k++)
            {
                int idx_calc_sobel = std::min(height - 1, std::max(0, idx_calc + k));
                int idx_temp_col = idx_calc_sobel % 3;
                int ratio = (k == 0) ? 2 : 1;
                double * STPtr = SobelTempMap.ptr<double>(idx_temp_col);
                for(int j = 0; j < width; j++)
                {
                    DPtr[j] += ratio * STPtr[j];
                }
            }
        }
    }
    double * DPtr = DMap.ptr<double>(height - 1);
    for(int k = -1; k <= 1; k++)
    {
        int idx_calc_sobel = std::min(height - 1, std::max(0, height - 1 + k));
        int idx_temp_col = idx_calc_sobel % 3;
        int ratio = (k == 0) ? 2 : 1;
        double * STPtr = SobelTempMap.ptr<double>(idx_temp_col);
        for(int j = 0; j < width; j++)
        {
            DPtr[j] += ratio * STPtr[j];
        }
    }

    // 未归一化的Dx与Dy的min和max：
    // -516.676 409.261
    // -472.163 380.63
}

int main()
{
    /* cv::Mat的索引和二维数组的索引在同一个坐标系下 */
    cv::Mat lena_img = cv::imread("D:\\MyFiles\\Year1a\\cv\\week2_edges\\lena_std.png");
    cv::Mat gray_img;
    cv::cvtColor(lena_img, gray_img, cv::COLOR_BGR2GRAY);
    if(gray_img.empty())
    {
        std::cout << "Cannot load image!!!" << std::endl;
        return -1;
    }

    int lena_height = gray_img.rows; // 竖边长度
    int lena_width = gray_img.cols; // 横边长度
    int lena_channels = gray_img.channels(); // 读成灰度值，单通道，1

    int wind_size;
    std::cout << "Define window size with one ODD int: ";
    std::cin >> wind_size;
    int half_wind_size = wind_size / 2;
    double sigma;
    std::cout << "Please enter the sigma value: ";
    std::cin >> sigma;
    std::cout << "Calculating Gaussian weights... " << std::endl;
    std::vector<double> GF_Weights(wind_size);
    double denom_no_PI = 2 * sigma * sigma;
    double sum1 = 0;
    for(int i = 0; i < wind_size; i++) 
    {
        double x2 = (i - half_wind_size) * (i - half_wind_size);
        double log_num = - x2 / (denom_no_PI);
        double num = exp(log_num);
        GF_Weights[i] = num;
        sum1 += GF_Weights[i];
    }   
    for(int i = 0; i < wind_size; i++)
    {
        GF_Weights[i] /= sum1;
    }

    // 二维高斯滤波
    cv::Mat TempMap(wind_size, lena_width, CV_64F, cv::Scalar(0));
    cv::Mat GaussMap(lena_height, lena_width, CV_64F, cv::Scalar(0));

    // Gemini建议的Lambda表达式，由于两块代码是一样的，因此可以让代码更美观
    // 此处实现的是高斯滤波的列计算
    auto calc_col_gauss = [&](int row)
    {
        int idx_calc_col = row; // 永远在计算half window之前的高斯，待计算的行
        double * GMapPtr = GaussMap.ptr<double>(idx_calc_col);
        for(int j = -half_wind_size; j <= half_wind_size; j++) // 换个思路，这次把列加权计算放外环
        {
            // idx_calc_col为本次列计算的行，idx_real为核计算涉及的行
            int idx_real = std::min(lena_height - 1, std::max(0, idx_calc_col + j)); 
            int idx_wind_col = idx_real % wind_size; // 临时表对应的行
            double * TMapPtr = TempMap.ptr<double>(idx_wind_col); // 如此可以多次重复地用一个TMapPtr
            double weight = GF_Weights[half_wind_size + j];
            for(int k = 0; k < lena_width; k++)
            {
                GMapPtr[k] += weight * TMapPtr[k];
            }
        }
    };

    for(int i = 0; i < lena_height; i++)
    {
        int idx_swin = i % wind_size; // 这么写是没有问题的，unlike ChatGPT suggestions
        uchar * GImgPtr = gray_img.ptr<uchar>(i);
        double * TMapPtr = TempMap.ptr<double>(idx_swin);
        // 行高斯计算，建临时表
        for(int j = 0; j < lena_width; j++)
        {
            TMapPtr[j] = 0; // 清除上一次的临时表数据
            for(int k = -half_wind_size; k <= half_wind_size; k++)
            {
                int idx_calc_row = std::max(0, std::min(lena_width - 1, j + k));
                TMapPtr[j] += GF_Weights[k + half_wind_size] * GImgPtr[idx_calc_row];
            }
        }
        // 列高斯计算，使用临时表
        if(i >= half_wind_size) // 需要未来的half window才可以开始计算
        {
            calc_col_gauss(i - half_wind_size);
        }
    }
    // 补最后的half window的高斯计算，该用的临时表其实已经建立好了
    for(int i = 0; i < half_wind_size; i++)
    {
        calc_col_gauss(lena_height - half_wind_size + i);
    }

    // 换新的思路：没必要分为0, 45, 90, 135，这4类
    // 而是通过比较Dx和Dy的大小关系，结合插值，进行下一个模块的NMS进程
    cv::Mat Magnitude(lena_height, lena_width, CV_64F);
    cv::Mat DxMap(lena_height, lena_width, CV_64F, cv::Scalar(0));
    cv::Mat DyMapTran(lena_width, lena_height, CV_64F, cv::Scalar(0));

    // 思路4：换成3x3 Sobel算子计算Dx和Dy
    // 坑自己的思路：横竖Sobel算子是转置关系，可以将lena转置再转置实现代码复用
    // 采用Dx的Sobel算子的第一行，[-1, 0, 1]；第三行1与之一致，第二行为2倍
    calc_sobel(lena_height, lena_width, DxMap, GaussMap);
    cv::Mat GaussMapTran = GaussMap.t();
    calc_sobel(lena_width, lena_height, DyMapTran, GaussMapTran);
    cv::Mat DyMap = DyMapTran.t();
    cv::sqrt(DxMap.mul(DxMap) + DyMap.mul(DyMap), Magnitude);
    cv::imwrite("D:\\MyFiles\\Year1a\\cv\\week2_edges\\lena_dx.jpg", DxMap);
    cv::imwrite("D:\\MyFiles\\Year1a\\cv\\week2_edges\\lena_dy.jpg", DyMap);
    cv::imwrite("D:\\MyFiles\\Year1a\\cv\\week2_edges\\lena_mag.jpg", Magnitude);

    // 计算NMS
    // 这里采用插值进行计算
    cv::Mat NMSMap(lena_height, lena_width, CV_64F, cv::Scalar(0));
    int x1, y1, x2, y2;
    for(int i = 0; i < lena_height; i++)
    {
        double * NMSPtr = NMSMap.ptr<double>(i);
        double * MagniPtr = Magnitude.ptr<double>(i);
        double * DxMPtr = DxMap.ptr<double>(i);
        double * DyMPtr = DyMap.ptr<double>(i);
        for(int j = 0; j < lena_width; j++)
        {
            double Dx_ij = DxMPtr[j];
            double Dy_ij = DyMPtr[j];
            double Mag_ij = MagniPtr[j];
            double mag_prev = 0;
            double mag_post = 0;
            double ratio = 0;
            // 从原版的4个角度分类问题，转换成：“Dx和Dy值比大小 + 插值 + NMS”的流程
            // 插值的目的其实是为了Magnitude的NMS相比四分类更加“客观公正”
            // 我认为不一定，但是确实看上去似乎合理些
            x1 = std::min(lena_height - 1, i + 1);
            x2 = std::max(0, i - 1);
            y1 = std::min(lena_width - 1, j + 1);
            y2 = std::max(0, j - 1);
            if(fabs(Dx_ij) > fabs(Dy_ij)) // 横向梯度为主导
            {
                ratio = fabs(Dy_ij / Dx_ij);
                if(Dx_ij * Dy_ij > 0) // 45度（以下和右为正方向）
                {
                    // ratio为0，即为0度情况；ratio为1，即为45/135度情况
                    mag_prev = (1 - ratio) * MagniPtr[y1] + ratio * Magnitude.at<double>(x1, y1);
                    mag_post = (1 - ratio) * MagniPtr[y2] + ratio * Magnitude.at<double>(x2, y2);
                }
                else // 135度
                {             
                    mag_prev = (1 - ratio) * MagniPtr[y1] + ratio * Magnitude.at<double>(x2, y1);
                    mag_post = (1 - ratio) * MagniPtr[y2] + ratio * Magnitude.at<double>(x1, y2);
                }
            }
            else // 纵向梯度为主导
            {
                ratio = fabs(Dx_ij / Dy_ij);
                if(Dx_ij * Dy_ij > 0) // 45度
                {
                    // ratio为0，即为90度情况；ratio为1，即为45/135度情况
                    mag_prev = (1 - ratio) * Magnitude.at<double>(x1, j) + ratio * Magnitude.at<double>(x1, y1);
                    mag_post = (1 - ratio) * Magnitude.at<double>(x2, j) + ratio * Magnitude.at<double>(x2, y2);
                }
                else // 135度
                {             
                    mag_prev = (1 - ratio) * Magnitude.at<double>(x1, j) + ratio * Magnitude.at<double>(x1, y2);
                    mag_post = (1 - ratio) * Magnitude.at<double>(x2, j) + ratio * Magnitude.at<double>(x2, y1);
                }
            }
            if(Mag_ij >= mag_prev && Mag_ij >= mag_post)
            {
                NMSPtr[j] = Mag_ij;
            }
        }
    }
    cv::imwrite("D:\\MyFiles\\Year1a\\cv\\week2_edges\\lena_nms.jpg", NMSMap);

    // 二值化与最终的边缘展示
    // ChatGPT建议把这个任务作为一个深度优先或广度优先的小任务
    // 思路：遍历所有的strong点，把周边的weak点全部变成strong点，再把新strong点遍历一遍，迭代
    // 问题是：如此这般，会不会只是把weak点全部变成strong点？
    // 孤立的weak点应该会在这个流程中被滤掉
    // 我需要两个容器，一个放旧strong点，一个放新strong点
    // 每次迭代中，旧strong点拿一个扔一个，迭代结束后新strong点全部填入旧strong点
    // 扔掉的点保留在cv::Mat中
    // ChatGPT的建议是：改value threshold为percentile threshold
    // 将NMSMap的值排序，取前90%为阈值值，需要用到std::sort或者std::nth_element
    // 因为value threshold和光照强度相关，而不是百分比分布
    // cv::findNonZero是个很有意思的函数，本意是用于CV_8U，
    // 1e-6都会被认为是是非零，对于CV_64F可以认为是个Point收集函数
    std::vector<cv::Point> nms_points;
    cv::findNonZero(NMSMap, nms_points);
    int nms_points_size_percentile = static_cast<int>(nms_points.size() * 0.72 - 1);
    std::nth_element(
        nms_points.begin(), 
        nms_points.begin() + nms_points_size_percentile, // 需要其升序
        nms_points.end(),
        [&] (cv::Point& a, cv::Point& b)
        {
            return (NMSMap.at<double>(a.y, a.x) < NMSMap.at<double>(b.y, b.x));
        }
    );
    cv::Point thesh_point = nms_points[nms_points_size_percentile];
    double HighThresh = NMSMap.at<double>(thesh_point.y, thesh_point.x);
    double LowThresh = 0.4 * HighThresh;

    // cv::Mat HighMap(lena_height, lena_width, CV_8U);
    // cv::Mat LowMap(lena_height, lena_width, CV_8U);
    // HighMap数据类型实际是CV_64F，即6；CV_8U为0
    // 测试下来发现，cv::threshold的dst数据类型和src保持一致，而不是遵循cv::Mat初始化时的定义
    // ChatGPT的解释是，cv::threshold等一系列其他cv函数，如cv::convertTo等，会在内部调用create函数
    // create会检测dst和src在大小、数据类型等分配上是否一致，一致则沿用，否则开辟新的空间
    // 所以按理说不应该初始化，直接定义接cv::threshold即可
    cv::Mat HighMap;
    cv::Mat LowMap;
    cv::threshold(NMSMap, HighMap, HighThresh, 256, cv::THRESH_BINARY); 
    cv::threshold(NMSMap, LowMap, LowThresh, 256, cv::THRESH_BINARY);
    std::cout << HighMap.type() << " " << LowMap.type() << std::endl;
    cv::imwrite("D:\\MyFiles\\Year1a\\cv\\week2_edges\\lena_nms_high.jpg", HighMap);
    cv::imwrite("D:\\MyFiles\\Year1a\\cv\\week2_edges\\lena_nms_low.jpg", LowMap);

    std::queue<int> strongPoints;
    for(int i = 0; i < lena_height; i++)
    {
        double * HighPtr = HighMap.ptr<double>(i); // Solved!!! 
        for(int j = 0; j < lena_width; j++)
        {
            if(HighPtr[j] > 1e-6)
            {
                strongPoints.push(i);
                strongPoints.push(j);
            }
        }
    }
    while(! strongPoints.empty())
    {
        int idx_i = strongPoints.front();
        strongPoints.pop();
        int idx_j = strongPoints.front();
        strongPoints.pop();

        double * HighMPtr = HighMap.ptr<double>(idx_i);
        HighMPtr[idx_j] = 255;
        double * LowMPtr = LowMap.ptr<double>(idx_i);
        LowMPtr[idx_j] = 0;
        for(int dx = -1; dx <= 1; dx++)
        {
            if(idx_i + dx < 0 || idx_i + dx >= lena_height)
            {
                continue;
            }
            double * LowPtr = LowMap.ptr<double>(idx_i + dx);
            for(int dy = -1; dy <= 1; dy++)
            {
                if(idx_j + dy < 0 || idx_j + dy >= lena_width)
                {
                    continue;
                }
                if(LowPtr[idx_j + dy] > 1e-6)
                {
                    strongPoints.push(idx_i + dx);
                    strongPoints.push(idx_j + dy);
                }
            }
        }
    }

    cv::imwrite("D:\\MyFiles\\Year1a\\cv\\week2_edges\\lena_edges.jpg", HighMap);
    cv::imwrite("D:\\MyFiles\\Year1a\\cv\\week2_edges\\lena_post_edges.jpg", LowMap);
    return 0;
}