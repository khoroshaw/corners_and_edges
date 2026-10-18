#include<iostream>
#include<cmath>
#include<opencv2/core.hpp>
#include<opencv2/imgproc.hpp>
#include<opencv2/imgcodecs.hpp>
#include<opencv2/highgui.hpp>
#include<vector>
#include<set>
#include<algorithm>
#include<cstring>
#include<string>

struct cvPointwithScale
{
    double corner_x;
    double corner_y;
    double scale_val;
};

void calcGaussian(int kernel_size, double sigma, cv::Mat& Image, cv::Mat& dst)
{
    int height = Image.rows; // 竖边长度
    int width = Image.cols; // 横边长度
    int channel_num = Image.channels(); // 默认使用灰度值，所以是1

    int half_size = static_cast<int>(kernel_size / 2); // 默认kernel_size是奇数
    std::vector<double> kernel_weights;
    double sum1 = 0;
    for(int k = -half_size; k <= half_size; k++)
    {
        double expo = -k * k / (2 * sigma * sigma);
        double weight = exp(expo);
        kernel_weights.push_back(weight);
        sum1 += weight;
    }
    for(int i = 0; i < kernel_size; i++)
    {
        kernel_weights[i] /= sum1;
    }
    
    cv::Mat RowCalcTemp(kernel_size, width, CV_64F, cv::Scalar(0));
    for(int i = 0; i < height; i++)
    {
        int idx_row_temp = i % kernel_size;
        double * RowPtr = RowCalcTemp.ptr<double>(idx_row_temp);
        double * ImgPtr = Image.ptr<double>(i);
        // 行计算
        for(int j = 0; j < width; j++)
        {
            RowPtr[j] = 0;
            for(int k = -half_size; k <= half_size; k++)
            {
                int idx_padding = std::min(width - 1, std::max(0, j + k));
                RowPtr[j] += kernel_weights[k + half_size] * ImgPtr[idx_padding];
            }
        }
        // 列计算
        if(i >= half_size)
        {
            int idx_col_dst = i - half_size;
            double * dstPtr = dst.ptr<double>(idx_col_dst);
            for(int k = -half_size; k <= half_size; k++)
            {
                int idx_col_calc = std::min(height - 1, std::max(0, idx_col_dst + k));
                int idx_col_temp = idx_col_calc % kernel_size;
                double * RowCalcPtr = RowCalcTemp.ptr<double>(idx_col_temp);
                int k_idx = k + half_size;
                for(int j = 0; j < width; j++)
                {
                    dstPtr[j] += kernel_weights[k_idx] * RowCalcPtr[j];
                }
            }
        }
    }
    for(int i = height - half_size; i < height; i++)
    {
        int idx_col_dst = i;
        double * dstPtr = dst.ptr<double>(idx_col_dst);
        for(int k = -half_size; k <= half_size; k++)
        {
            int idx_col_calc = std::min(height - 1, std::max(0, idx_col_dst + k));
            int idx_col_temp = idx_col_calc % kernel_size;
            double * RowCalcPtr = RowCalcTemp.ptr<double>(idx_col_temp);
            int k_idx = k + half_size;
            for(int j = 0; j < width; j++)
            {
                dstPtr[j] += kernel_weights[k_idx] * RowCalcPtr[j];
            }
        }
    }
}

void calcSigmas(int oct_num, int img_num, double sigma, double ratio, std::vector<std::vector<double>>& sigma_vectors)
{
    double sigma_mat_i_0 = sigma;
    for(int i = 0; i < oct_num; i++)
    {
        std::vector<double> sigma_value_octave;
        double sigma_mat_i_j = sigma_mat_i_0;
        for(int j = 0; j < img_num; j++)
        {
            sigma_value_octave.push_back(sigma_mat_i_j);
            sigma_mat_i_j *= ratio; // 一个octave中的每张DoG之间不同的sigma，差一个ratio
        }
        sigma_vectors.push_back(sigma_value_octave);
        sigma_mat_i_0 *= 2;
    }
}

void calcGaussiansPerOctave(cv::Mat& Img0, std::vector<double>& sigma_vector, std::vector<cv::Mat>& gauss_vector)
{
    int height = Img0.rows; // 竖边长度
    int width = Img0.cols; // 横边长度
    int channel_num = Img0.channels(); // 默认使用灰度值，所以是1

    for(auto& sigma_ptr : sigma_vector)
    {
        cv::Mat Img_Gaussed(height, width, CV_64F, cv::Scalar(0));
        // ChatGPT的建议有问题，但是方向是对的
        // 原本我一直设定kernel是3x3，太小了，根本看不出sigma维度后的区别
        int kernel_size = static_cast<int>(sqrt(std::min(height, width)));
        kernel_size = (kernel_size % 2 == 0) ? kernel_size - 1 : kernel_size;
        calcGaussian(kernel_size, sigma_ptr, Img0, Img_Gaussed);
        // std::cout << sigma_ptr << ", ";
        gauss_vector.push_back(Img_Gaussed.clone()); // 否则为浅拷贝，会是一行一样的图
    }
    // std::cout << std::endl;
}

void calcAllGaussians(cv::Mat& Img0, int oct_num, std::vector<std::vector<double>>& sigma_vectors, std::vector<std::vector<cv::Mat>>& gauss_vectors)
{
    int height = Img0.rows; // 竖边长度
    int width = Img0.cols; // 横边长度
    int channel_num = Img0.channels(); // 默认使用灰度值，所以是1

    int gauss_img_num = sigma_vectors[0].size();

    // Img0是灰度图像，数据类型是8U，而我的函数都是64F
    // 因此需要将Img0转换成64F
    cv::Mat Img0_64F;
    Img0.convertTo(Img0_64F, CV_64F);

    for(int i = 0; i < oct_num; i++)
    {
        std::vector<cv::Mat> gauss_imgs;
        if(i == 0)
        {
            calcGaussiansPerOctave(Img0_64F, sigma_vectors[i], gauss_imgs);
        }
        else
        {
            cv::Mat Img_i = gauss_vectors[i - 1][gauss_img_num - 3];
            cv::Mat Img_i_downsampled;
            cv::pyrDown(Img_i, Img_i_downsampled); // 默认下采样1/2
            calcGaussiansPerOctave(Img_i_downsampled, sigma_vectors[i], gauss_imgs);
        }
        gauss_vectors.push_back(gauss_imgs);
    }
}

void collectAllPertientPointersPerRow(std::vector<cv::Mat>& DoGs_vector, int idx_dog, int idx_row, std::vector<double *>& DoGMapRowPtrs)
{
    int height = DoGs_vector[0].rows; // 竖边长度
    int width = DoGs_vector[0].cols; // 横边长度
    int channels_num = DoGs_vector[0].channels(); // 读成灰度值，单通道，1
    int dogs_num = DoGs_vector.size();

    std::vector<cv::Mat> DoGs_vector_temp;
    DoGs_vector_temp.push_back(DoGs_vector[idx_dog]);
    if(idx_dog > 0) // 3x3Scale遍历
    {
        DoGs_vector_temp.push_back(DoGs_vector[idx_dog - 1]);
    }
    if(idx_dog < dogs_num - 1)
    {
        DoGs_vector_temp.push_back(DoGs_vector[idx_dog + 1]);
    }

    for(auto& DoGMap_idx : DoGs_vector_temp) // 3x3行遍历
    {
        double * DoGMapRowPtr_pres = DoGMap_idx.ptr<double>(idx_row);
        DoGMapRowPtrs.push_back(DoGMapRowPtr_pres);
        if(idx_row > 0)
        {
            double * DoGMapRowPtr_prev = DoGMap_idx.ptr<double>(idx_row - 1);
            DoGMapRowPtrs.push_back(DoGMapRowPtr_prev);
        }
        if(idx_row < height - 1)
        {
            double * DoGMapRowPtr_post = DoGMap_idx.ptr<double>(idx_row + 1);
            DoGMapRowPtrs.push_back(DoGMapRowPtr_post);
        }
    }
}

void findExtrema26PerRow(std::vector<double *>& DoGMapRowPtrs, int idx_dog, int idx_row, int width, std::vector<cvPointwithScale>& corners_vector)
{
    // DoGMapRowPtrs[0]为目前点所在行的索引，其他的都不是
    int ptrs_num = DoGMapRowPtrs.size();
    for(int j = 0; j < width; j++)
    {
        double center_val = DoGMapRowPtrs[0][j]; // center_val需要是26近邻中最大或者最小点
        double min_26_val = center_val;
        double max_26_val = center_val;
        for(auto& dog_ptr : DoGMapRowPtrs)
        {
            for(int dj = -1; dj <= 1; dj++) // 3x3列遍历
            {
                if(j + dj < 0 || j + dj > width - 1)
                {
                    continue;
                }
                min_26_val = std::min(min_26_val, dog_ptr[j + dj]);
                max_26_val = std::max(max_26_val, dog_ptr[j + dj]);
            }
        }
        if(min_26_val != center_val && max_26_val != center_val)
        {
            continue;
        }
        else
        {
            cvPointwithScale point_and_scale;
            point_and_scale.corner_x = static_cast<double>(idx_row);
            point_and_scale.corner_y = static_cast<double>(j);
            point_and_scale.scale_val = static_cast<double>(idx_dog);
            corners_vector.push_back(point_and_scale);
        }
    }
}

void findCornersPerOctave(std::vector<cv::Mat>& dogs_vector, std::vector<cvPointwithScale>& corners_vector)
{
    // 26近邻
    int height = dogs_vector[0].rows; // 竖边长度
    int width = dogs_vector[0].cols; // 横边长度
    int channels_num = dogs_vector[0].channels(); // 读成灰度值，单通道，1
    int dog_num = dogs_vector.size();

    for(int idx_dog = 0; idx_dog < dog_num; idx_dog++)
    {
        for(int idx_row = 0; idx_row < height; idx_row++)
        {
            std::vector<double *> DoGMapRowPtrs; // 26近邻所需的全部pointers
            collectAllPertientPointersPerRow(dogs_vector, idx_dog, idx_row, DoGMapRowPtrs);
            // std::cout << "collectAllPertientPointersPerRow: " << idx_dog << ", " << idx_row << std::endl;
            // DoGMapRowPtrs[0]为待定行，其他都是近邻行
            findExtrema26PerRow(DoGMapRowPtrs, idx_dog, idx_row, width, corners_vector);
            // std::cout << "findExtrema26PerRow: " << idx_dog << ", " << idx_row << std::endl;
        }
    }
}

void calcEachExtremaShift(std::vector<cv::Mat>& dogs_vector, std::vector<cvPointwithScale>& corners_vector, double dx_thresh, double r_thresh, std::vector<cvPointwithScale>& fit_corners_vector)
{
    // 涉及梯度和海塞阵的计算，我不打算全算，而是来一个点计算一套出来
    int dog_num = dogs_vector.size();
    int height = dogs_vector[0].rows;
    int width = dogs_vector[0].cols;
    int channels_num = dogs_vector[0].channels();
    for(auto& original_corner_with_scale_i : corners_vector)
    { 
        auto corner_with_scale_i = original_corner_with_scale_i; // ChatGPT提示不要再for循环中更改它
        cv::Mat deltaX;
        for(int cnt = 0; cnt < 5; cnt++)
        {
            // 本点的相关数据
            int dog_idx = static_cast<int>(corner_with_scale_i.scale_val);
            int corner_point_x = static_cast<int>(corner_with_scale_i.corner_x);
            int corner_point_y = static_cast<int>(corner_with_scale_i.corner_y);
            // 其他相关值
            cv::Mat DoGMap = dogs_vector[dog_idx];
            cv::Mat DoGMapPost = dogs_vector[std::min(dog_num - 1, dog_idx + 1)];
            cv::Mat DoGMapPrev = dogs_vector[std::max(0, dog_idx - 1)];
            // 计算梯度和海塞阵涉及的行
            double * D_i_s = DoGMap.ptr<double>(corner_point_x);
            double * D_1i_s = DoGMap.ptr<double>(std::max(0, corner_point_x - 1));
            double * D_i1_s = DoGMap.ptr<double>(std::min(height - 1, corner_point_x + 1));
            double * D_i_s1 = DoGMapPost.ptr<double>(corner_point_x);
            double * D_1i_s1 = DoGMapPost.ptr<double>(std::max(0, corner_point_x - 1));
            double * D_i1_s1 = DoGMapPost.ptr<double>(std::min(height - 1, corner_point_x + 1));
            double * D_i_1s = DoGMapPrev.ptr<double>(corner_point_x);
            double * D_1i_1s = DoGMapPrev.ptr<double>(std::max(0, corner_point_x - 1));
            double * D_i1_1s = DoGMapPrev.ptr<double>(std::min(height - 1, corner_point_x + 1));
            // 计算梯度和海塞阵涉及的值
            double D_i_j_s = D_i_s[corner_point_y];
            double D_i1_j_s = D_i1_s[corner_point_y];
            double D_1i_j_s = D_1i_s[corner_point_y];
            double D_i_j1_s = D_i_s[std::min(width - 1, corner_point_y + 1)];
            double D_i_1j_s = D_i_s[std::max(0, corner_point_y - 1)];
            double D_i_j_1s = D_i_1s[corner_point_y];
            double D_i_j_s1 = D_i_s1[corner_point_y];
            double D_i1_j1_s = D_i1_s[std::min(width - 1, corner_point_y + 1)];
            double D_i1_1j_s = D_i1_s[std::max(0, corner_point_y - 1)];
            double D_1i_j1_s = D_1i_s[std::min(width - 1, corner_point_y + 1)];
            double D_1i_1j_s = D_1i_s[std::max(0, corner_point_y - 1)];
            double D_i_j1_s1 = D_i_s1[std::min(width - 1, corner_point_y + 1)];
            double D_i_1j_s1 = D_i_s1[std::max(0, corner_point_y - 1)];
            double D_i_j1_1s = D_i_1s[std::min(width - 1, corner_point_y + 1)];
            double D_i_1j_1s = D_i_1s[std::max(0, corner_point_y - 1)];
            double D_i1_j_s1 = D_i1_s1[corner_point_y];
            double D_1i_j_s1 = D_1i_s1[corner_point_y];
            double D_i1_j_1s = D_i1_1s[corner_point_y];
            double D_1i_j_1s = D_1i_1s[corner_point_y];
            // 计算梯度
            double dD_dx = (D_i1_j_s - D_1i_j_s) / 2.0;
            double dD_dy = (D_i_j1_s - D_i_1j_s) / 2.0;
            double dD_ds = (D_i_j_s1 - D_i_j_1s) / 2.0;
            // 计算海塞阵
            // 计算二阶导，可以套用一阶导的公式，但这样会涉及诸如“x + 2”索引的值
            // SIFT希望能够将涉及计算的值固定在26近邻范围内，因此做了近似
            double dD_dxdx = D_i1_j_s + D_1i_j_s - 2.0 * D_i_j_s;
            double dD_dydy = D_i_j1_s + D_i_1j_s - 2.0 * D_i_j_s;
            double dD_dsds = D_i_j_s1 + D_i_j_1s - 2.0 * D_i_j_s;
            double dD_dxdy = 0.25 * (D_i1_j1_s - D_i1_1j_s - D_1i_j1_s + D_1i_1j_s);
            double dD_dyds = 0.25 * (D_i_j1_s1 - D_i_1j_s1 - D_i_j1_1s + D_i_1j_1s);
            double dD_dxds = 0.25 * (D_i1_j_s1 - D_1i_j_s1 - D_i1_j_1s + D_1i_j_1s);
            // 解线性方程
            cv::Mat H(3, 3, CV_64F, cv::Scalar(0)); 
            std::vector<double> H_vals = {dD_dxdx, dD_dxdy, dD_dxds, dD_dxdy, dD_dydy, dD_dyds, dD_dxds, dD_dyds, dD_dsds};  
            cv::Mat dDdX(3, 1, CV_64F, cv::Scalar(0));
            std::vector<double> dDdX_vals = {dD_dx, dD_dy, dD_ds};  
            for(int i = 0; i < 3; i++)
            {
                for(int j = 0; j < 3; j++)
                {
                    H.at<double>(i, j) = H_vals[i * 3 + j];
                }
                dDdX.at<double>(i, 0) = dDdX_vals[i];
            }
            cv::solve(H, -dDdX, deltaX, cv::DECOMP_LU); 
            cv::Mat deltaXAbs = cv::abs(deltaX);
            bool no_greater_05 = true;
            for(int i = 0; i < 3; i++)
            {
                double deltaX_i_sgn = static_cast<double>((deltaX.at<double>(i, 0) > 0) - (deltaX.at<double>(i, 0) < 0));
                if(deltaXAbs.at<double>(i, 0) > 0.5)
                {
                    no_greater_05 = false;
                    // 需要保证这三个在下一次循环中是有意义的索引！
                    if(i == 0)
                    {
                       corner_with_scale_i.corner_x += deltaX_i_sgn; 
                    }
                    else if(i == 1)
                    {
                       corner_with_scale_i.corner_y += deltaX_i_sgn;
                    }
                    else if(i == 2)
                    {
                       corner_with_scale_i.scale_val += deltaX_i_sgn;
                    }
                }
            }
            if(no_greater_05)
            {
                // 对比度低的点
                double Dx_hat = D_i_j_s + 0.5 * (
                    dD_dx * deltaX.at<double>(0, 0) +
                    dD_dy * deltaX.at<double>(1, 0) +
                    dD_ds * deltaX.at<double>(2, 0)
                );
                // 边缘检测点
                double Trace = dD_dxdx + dD_dydy;
                double Det = dD_dxdx * dD_dydy - dD_dxdy * dD_dxdy;
                double trace2_det = Trace * Trace / Det;
                double r_ratio_thresh = (r_thresh + 1) * (r_thresh + 1) / r_thresh;
                if(fabs(Dx_hat) < dx_thresh || trace2_det >= r_ratio_thresh) // 去除对比度低的点和边缘检测点
                {
                    corner_with_scale_i.corner_x = -1;
                }
                else
                {
                    corner_with_scale_i.corner_x += deltaX.at<double>(0, 0);
                    corner_with_scale_i.corner_y += deltaX.at<double>(1, 0);
                    corner_with_scale_i.scale_val += deltaX.at<double>(2, 0);
                }
                break;
            }
            bool x_in_bound = (corner_with_scale_i.corner_x > 0) && (corner_with_scale_i.corner_x < height - 1);
            bool y_in_bound = (corner_with_scale_i.corner_y > 0) && (corner_with_scale_i.corner_y < width - 1);
            bool s_in_bound = (corner_with_scale_i.scale_val > 0) && (corner_with_scale_i.scale_val < dog_num - 1);
            if(!x_in_bound || !y_in_bound || !s_in_bound)
            {
                break;
            }
        }
        bool x_in_bound = (corner_with_scale_i.corner_x > 0) && (corner_with_scale_i.corner_x < height - 1);
        bool y_in_bound = (corner_with_scale_i.corner_y > 0) && (corner_with_scale_i.corner_y < width - 1);
        bool s_in_bound = (corner_with_scale_i.scale_val > 0) && (corner_with_scale_i.scale_val < dog_num - 1);
        bool dx_in_bound = (deltaX.at<double>(0, 0) < 0.5) && (deltaX.at<double>(0, 0) > -0.5);
        bool dy_in_bound = (deltaX.at<double>(1, 0) < 0.5) && (deltaX.at<double>(1, 0) > -0.5);
        bool ds_in_bound = (deltaX.at<double>(2, 0) < 0.5) && (deltaX.at<double>(2, 0) > -0.5);
        if(x_in_bound && y_in_bound && s_in_bound && dx_in_bound && dy_in_bound && ds_in_bound) // 排除这个点的条件
        {
            fit_corners_vector.push_back(corner_with_scale_i);
        }
    }
}

int main()
{
    // 读入图像并转成灰度图
    cv::Mat lena_img, gray_img;
    lena_img = cv::imread("D:\\MyFiles\\Year1a\\cv\\week2_edges\\lena_std.png");
    cv::cvtColor(lena_img, gray_img, cv::COLOR_BGR2GRAY);
    if(gray_img.empty())
    {
        std::cout << "Cannot load image!!!" << std::endl;
        return -1;
    }

    // 计算长宽通道数
    int lena_height = gray_img.rows; // 竖边长度
    int lena_width = gray_img.cols; // 横边长度
    int lena_channels = gray_img.channels(); // 读成灰度值，单通道，1
 
    // 计算金字塔层数
    int t = 3; // t ∈ [0, log2(std::min(lena_height, lena_width))]
    int octave_num = static_cast<int>(log2(std::min(lena_height, lena_width)) - t);

    // 计算DoG的相关参数
    int s = 4; // 极值检测需要的img数，s ∈ [3, 5] 
    int s1 = s + 3; // 每层的img数
    double sigma_0 = 1.6 ; // 默认为1.6
    double k_expo = 1.0 / s;
    double k_ratio = pow(2.0, k_expo);

    // 计算每层，以及每张img的sigma值
    std::vector<std::vector<double>> sigma_value_double_vector; // [octave_num, s1]
    std::cout << "Calculating sigmas..." << std::endl;
    calcSigmas(octave_num, s1, sigma_0, k_ratio, sigma_value_double_vector);
    std::cout << "Sigmas calculation complete." << std::endl;

    // 计算金字塔
    std::vector<std::vector<cv::Mat>> all_octaves_vector;
    std::cout << "Calculating Gaussian filtered images..." << std::endl;
    calcAllGaussians(gray_img, octave_num, sigma_value_double_vector, all_octaves_vector);
    std::cout << "Gaussian filtered images calculation complete." << std::endl;

    // 计算金字塔差值，即DoG
    std::vector<std::vector<cv::Mat>> octave_dogs_vector;
    std::cout << "Calculating DoG images..." << std::endl;
    for(int i = 0; i < octave_num; i++) // 9 - 3 = 6
    {
        std::vector<cv::Mat> octave_dogs_i;
        for(int j = 1; j < s1; j++) // 4 + 3 - 1 = 6
        {
            octave_dogs_i.push_back(all_octaves_vector[i][j] - all_octaves_vector[i][j - 1]);
        }
        octave_dogs_vector.push_back(octave_dogs_i);
    }
    std::cout << "DoG images calculation complete." << std::endl;

    // 每个octave找26近邻最小点，并同时开始精确定位各个点
    std::vector<std::vector<cvPointwithScale>> corners_vector_vector;
    double dx_thresh = 3; // 判断对比度小的点，原版代码是0.03；dx_thresh越大，条件越苛刻
    double r_thresh = 7; // 判断边缘点，原版代码是10；r_thresh越小，条件越苛刻
    std::cout << "Octaves in total: " << octave_num << std::endl;
    for(int i = 0; i < octave_num; i++)
    {
        std::vector<cvPointwithScale> corners_vector;
        std::vector<cvPointwithScale> fit_corners_vector;
        std::vector<cv::Mat> dogs_octave_i = octave_dogs_vector[i];
        std::cout << "Finding interest points for the octave indexed " << i << "..." << std::endl;
        findCornersPerOctave(dogs_octave_i, corners_vector);
        std::cout << "Found " << corners_vector.size() << " interest points in octave " << i << "." << std::endl;
        // 定位各个点，计算偏移量，并加在原来的点中
        std::cout << "Localizing interest points with quadratic fit..." << std::endl;
        calcEachExtremaShift(dogs_octave_i, corners_vector, dx_thresh, r_thresh, fit_corners_vector);
        std::cout << "Localization complete. " << fit_corners_vector.size() << " interest points left." << std::endl;
        // 传入统计vector
        corners_vector_vector.push_back(fit_corners_vector);
    }

    // 整合所有octave的特征点
    std::vector<cvPointwithScale> up_fit_corners_vector;
    std::cout << "Re-organizing interest points from all octaves..." << std::endl;
    double up_ratio = 1.0;
    for(int i = 0; i < octave_num; i++)
    {
        for(auto& original_down_corner_scale : corners_vector_vector[i])
        {
            auto down_corner_scale = original_down_corner_scale;
            down_corner_scale.corner_x *= up_ratio;
            down_corner_scale.corner_y *= up_ratio;
            down_corner_scale.scale_val = sigma_0 * up_ratio * pow(k_ratio, down_corner_scale.scale_val);
            up_fit_corners_vector.push_back(down_corner_scale);
            cv::circle(lena_img, cv::Point(down_corner_scale.corner_y, down_corner_scale.corner_x), 5, cv::Scalar(0, 0, 255), 2);
        }
        up_ratio *= 2.0;
    }
    std::cout << "Re-organization complete. " << up_fit_corners_vector.size() << " interest points in total." << std::endl;

    cv::imwrite("D:\\MyFiles\\Year1a\\cv\\week2_edges\\lena_circled_sift.jpg", lena_img);
    return 0;
}