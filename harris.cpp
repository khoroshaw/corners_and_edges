#include<iostream>
#include<cmath>
#include<opencv2/core.hpp>
#include<opencv2/imgproc.hpp>
#include<opencv2/imgcodecs.hpp>
#include<opencv2/highgui.hpp>
#include<vector>
#include<algorithm>

void calc_grad_sobel(int height, int width, cv::Mat& Image, cv::Mat& Intensity)
{
    // 3x3的Sobel算子，用横向[-1, 0, 1]实现两个intensity的计算
    // 用行列两次计算实现二维算子
    // 下和右为正方向
    cv::Mat SobelTempMap(3, width, CV_64F);
    for(int i = 0; i < height; i++)
    {
        double * STPtr = SobelTempMap.ptr<double>(i % 3);
        uchar * ImgPtr = Image.ptr<uchar>(i); // 灰度图，uchar
        for(int j = 0; j < width; j++) // 行计算
        {
            STPtr[j] = 0;
            for(int k = -1; k <= 1; k = k + 2)
            {
                int idx_row_calc = std::min(width - 1, std::max(0, j + k));
                STPtr[j] += k * ImgPtr[idx_row_calc];
            }
        }
        if(i > 0)
        {
            int idx_col_calc = i - 1; // 列计算的行
            double * IntenPtr = Intensity.ptr<double>(idx_col_calc);
            for(int k = -1; k <= 1; k++) // 列计算
            {
                int idx_sobel_calc = std::min(height - 1, std::max(0, idx_col_calc + k)); // 算子的行
                int idx_tempmap_calc = idx_sobel_calc % 3; // 缓存区的行
                double * STMPtr = SobelTempMap.ptr<double>(idx_tempmap_calc);
                int ratio = (k == 0) ? 2 : 1; // Sobel
                for(int j = 0; j < width; j++)
                {
                    IntenPtr[j] += ratio * STMPtr[j];
                }
            }
        }
    }
    int idx_col_ult = height - 1; // 列计算的行
    double * IntenUPtr = Intensity.ptr<double>(idx_col_ult);
    for(int k = -1; k <= 1; k++) // 列计算
    {
        int idx_sobel_ult = std::min(height - 1, std::max(0, idx_col_ult + k)); // 算子的行
        int idx_tempmap_ult = idx_sobel_ult % 3; // 缓存区的行
        double * STMUPtr = SobelTempMap.ptr<double>(idx_tempmap_ult);
        int ratio = (k == 0) ? 2 : 1; // Sobel
        for(int j = 0; j < width; j++)
        {
            IntenUPtr[j] += ratio * STMUPtr[j];
        }
    }
}

int main()
{
    /* Hyper Parameters*/
    double K = 0.05;
    int nms_padding = 7;
    
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

    // 没必要储存Ix和Iy
    // 本算法用水平和竖直两个高斯核实现二维高斯滤波，以下为滤波时需要计算的值
    // 考虑尝试改用3x3的Sobel算子实现梯度计算，同时实现三个平方值的计算
    cv::Mat Ix(lena_height, lena_width, CV_64F);
    cv::Mat IyTran(lena_width, lena_height, CV_64F); 
    cv::Mat gray_img_tran = gray_img.t();
    calc_grad_sobel(lena_height, lena_width, gray_img, Ix);
    calc_grad_sobel(lena_width, lena_height, gray_img_tran, IyTran);
    cv::Mat Iy = IyTran.t();
    cv::imwrite("D:\\MyFiles\\Year1a\\cv\\week2_edges\\lena_ix.jpg", Ix);
    cv::imwrite("D:\\MyFiles\\Year1a\\cv\\week2_edges\\lena_iy.jpg", Iy);

    cv::Mat Ix2 = Ix.mul(Ix);
    cv::Mat Iy2 = Iy.mul(Iy);
    cv::Mat IxIy = Ix.mul(Iy);
    cv::imwrite("D:\\MyFiles\\Year1a\\cv\\week2_edges\\lena_ix2.jpg", Ix2);
    cv::imwrite("D:\\MyFiles\\Year1a\\cv\\week2_edges\\lena_iy2.jpg", Iy2);
    cv::imwrite("D:\\MyFiles\\Year1a\\cv\\week2_edges\\lena_ixiy.jpg", IxIy);

    int wind_size;
    double sigma;
    std::cout << "Define window size with one ODD int: ";
    std::cin >> wind_size;    
    int half_wind_size = wind_size / 2;
    std::cout << "Please enter the sigma value: ";
    std::cin >> sigma;
    
    std::cout << "Calculating Gaussian weights... " << std::endl;
    std::vector<double> Kernel_Weights_1d(wind_size); // 一维高斯实现二维卷积，降低时间复杂度
    double denom_no_PI = 2 * sigma * sigma;
    double sum1 = 0;
    for(int i = 0; i < wind_size; i++) 
    {
        double x2 = pow(i - wind_size / 2, 2);
        double log_num = - x2 / denom_no_PI;
        Kernel_Weights_1d[i] = exp(log_num);
        sum1 += Kernel_Weights_1d[i];
    }   
    for(int i = 0; i < wind_size; i++) 
    {
        Kernel_Weights_1d[i] /= sum1;
    }   

    // 计算GF与R
    cv::Mat M11Map(lena_height, lena_width, CV_64F, cv::Scalar(0)); 
    cv::Mat M12Map(lena_height, lena_width, CV_64F, cv::Scalar(0)); 
    cv::Mat M22Map(lena_height, lena_width, CV_64F, cv::Scalar(0)); 
    cv::Mat RMap(lena_height, lena_width, CV_64F, cv::Scalar(0));

    // lambda表达式
    auto calc_gaussian_filter = [&](cv::Mat& MMap, cv::Mat& IMap)
    {
        cv::Mat MTempMap(wind_size, lena_width, CV_64F, cv::Scalar(0));
 
        for(int i = 0; i < lena_height; i++)
        {
            double * MTMPtr = MTempMap.ptr<double>(i % wind_size);
            double * IMPtr = IMap.ptr<double>(i);
            // 计算行权相加 
            for(int j = 0; j < lena_width; j++)
            {
                MTMPtr[j] = 0;
                for(int k = -half_wind_size; k <= half_wind_size; k++)
                {
                    int idx_calc_row = std::min(lena_width - 1, std::max(0, j + k));
                    MTMPtr[j] += Kernel_Weights_1d[k + half_wind_size] * IMPtr[idx_calc_row];
                }
            }
            // 计算列权相加
            if(i > half_wind_size - 1)
            {
                int idx_M_row = i - half_wind_size; // 所计算的M的行索引
                double * MMPtr = MMap.ptr<double>(idx_M_row);
                for(int k = -half_wind_size; k <= half_wind_size; k++)
                {
                    int idx_Kernel_row = std::min(lena_height - 1, std::max(0, idx_M_row + k)); // 参与计算的GF的行索引
                    int idx_calc = idx_Kernel_row % wind_size; // 缓存区的行索引
                    double weight = Kernel_Weights_1d[k + half_wind_size];
                    double * MTempMPtr = MTempMap.ptr<double>(idx_calc);
                    for(int j = 0; j < lena_width; j++)
                    {
                        MMPtr[j] += weight * MTempMPtr[j];
                    }
                }
            }
        }
        for(int i = lena_height - half_wind_size; i < lena_height; i++)
        {
            int idx_M_row = i; // 所计算的M的行索引
            double * MMPtr = MMap.ptr<double>(idx_M_row);
            for(int k = -half_wind_size; k <= half_wind_size; k++)
            {
                int idx_Kernel_row = std::min(lena_height - 1, std::max(0, idx_M_row + k)); // 参与计算的GF的行索引
                int idx_calc = idx_Kernel_row % wind_size; // 缓存区的行索引
                double weight = Kernel_Weights_1d[k + half_wind_size];
                double * MTempMPtr = MTempMap.ptr<double>(idx_calc);
                for(int j = 0; j < lena_width; j++)
                {
                    MMPtr[j] += weight * MTempMPtr[j];
                }
            }
        }
    };

    calc_gaussian_filter(M11Map, Ix2);
    calc_gaussian_filter(M22Map, Iy2);
    calc_gaussian_filter(M12Map, IxIy);
    RMap = M11Map.mul(M22Map) - M12Map.mul(M12Map) - K * (M11Map + M22Map).mul(M11Map + M22Map);
    cv::imwrite("D:\\MyFiles\\Year1a\\cv\\week2_edges\\lena_m11.jpg", M11Map);
    cv::imwrite("D:\\MyFiles\\Year1a\\cv\\week2_edges\\lena_m22.jpg", M22Map);
    cv::imwrite("D:\\MyFiles\\Year1a\\cv\\week2_edges\\lena_m12.jpg", M12Map);
    cv::imwrite("D:\\MyFiles\\Year1a\\cv\\week2_edges\\lena_r.jpg", RMap);

    // 利用R判断角点
    // percentile thresholding
    std::vector<cv::Point> r_points;
    cv::findNonZero(RMap, r_points);
    int percentile_thresh = static_cast<int>(r_points.size() * 0.9 - 1);
    std::nth_element(
        r_points.begin(),
        r_points.begin() + percentile_thresh,
        r_points.end(),
        [&] (cv::Point& a, cv::Point& b)
        {
            return (RMap.at<double>(a.y, a.x) < RMap.at<double>(b.y, b.x));
        }
    );
    cv::Point threshold_point = r_points[percentile_thresh];
    double threshold_val = RMap.at<double>(threshold_point.y, threshold_point.x);
    // 找角点
    cv::Mat RThreshMap;
    cv::threshold(RMap, RThreshMap, threshold_val, 255, cv::THRESH_BINARY);
    std::vector<cv::Point> edge_points;
    cv::findNonZero(RThreshMap, edge_points);
    int edge_points_num = edge_points.size();
    int edge_points_nms = edge_points_num * 0.005; // 可以是其他比率
    // std::cout << edge_points_num << std::endl; // 26215
    
    bool use_anms = false;
    bool use_nms = true;
    if(use_anms) // applying adaptive nms
    {
        // anms实现思路：
        // 把edge_points升序排列，遍历，逐个比较，可以得到一个radii vector
        // 假使能够实现edge_points依据radii vector的降序排序，便可以得到nms的结果
        // 这个思路有问题：i should find the minimum radius instead of the minimum R response
        // Rj - c * Ri，这个差值最小，并不能代表，半径最小；这里我把两个R搞混了！！！
        double c_ratio = 0.9; // anms: min(Rj > c * Ri)
        struct point_and_radius
        {
            cv::Point point;
            double radius;
        };
        std::vector<point_and_radius> radii_points;
        // 升序还是要的，但是需要多一个打擂台
        // ChatGPT指出，c_ratio的存在，使得升序在逻辑上有问题 
        for(int i = 0; i < edge_points_num; i++) // 升序，最后一个点肯定是inf，本次循环不涉及它
        {
            point_and_radius poi_radi;
            poi_radi.point = edge_points[i];
            double R_i = RMap.at<double>(poi_radi.point.y, poi_radi.point.x);
            double radius_R = 1.79769e+30; // inf 1.79769e+308
            for(int j = 0; j < edge_points_num; j++)
            {
                if(j == i)
                {
                    continue;
                }
                double R_j = RMap.at<double>(edge_points[j].y, edge_points[j].x);
                if(R_j - c_ratio * R_i > 1e-6)
                {
                    double radius_temp = norm(poi_radi.point - edge_points[j]);
                    if(radius_temp < radius_R)
                    {
                        radius_R = radius_temp;
                    }
                }
            }
            poi_radi.radius = radius_R;
            radii_points.push_back(poi_radi);
        }
        std::nth_element(
            radii_points.begin(),
            radii_points.begin() + edge_points_nms,
            radii_points.end(),
            [&] (point_and_radius& a, point_and_radius& b)
            {
                return (a.radius > b.radius); // 降序
            }
        );
        for(int i = 0; i < edge_points_nms; i++)
        {
            cv::circle(lena_img, radii_points[i].point, 5, cv::Scalar(0, 0, 255), 2);
        }
    }
    else if(use_nms) // local maximum nms
    {
        // assuming 3x3 nms window
        for(int i = 0; i < edge_points_num; i++)
        {
            // 遍历edge_points每个点，查看其RMap中3x3近邻，一旦出现response值比之大的就退出
            int idx_x = edge_points[i].x;
            int idx_y = edge_points[i].y;
            double R_now = RMap.at<double>(edge_points[i]);
            bool is_max = true;
            for(int dx = -1; dx <= 1; dx++)
            {
                for(int dy = -1; dy <= 1; dy++)
                {
                    if(dx == 0 && dy == 0)
                    {
                        continue;
                    }
                    int idx_x_n = std::min(lena_height - 1, std::max(0, idx_x + dx));
                    int idx_y_n = std::min(lena_width - 1, std::max(0, idx_y + dy));
                    double R_neighbor = RMap.at<double>(idx_y_n, idx_x_n);
                    if(R_neighbor - R_now > 1e-6)
                    {
                        is_max = false;
                        break;
                    }
                }
            }
            if(is_max)
            {
                cv::circle(lena_img, edge_points[i], 5, cv::Scalar(0, 0, 255), 2);
            }
        }
    }
    else
    {
        for(const auto& point_ptr : edge_points) 
        {
            cv::circle(lena_img, point_ptr, 5, cv::Scalar(0, 0, 255), 2);
        }
    }

    cv::imwrite("D:\\MyFiles\\Year1a\\cv\\week2_edges\\lena_edge_points.jpg", lena_img);
    return 0;
}