#include<iostream>
#include<cmath>
#include<opencv2/core.hpp>
#include<opencv2/imgproc.hpp>
#include<opencv2/imgcodecs.hpp>
#include<opencv2/highgui.hpp>
#include<vector>
#include<algorithm>
#include<cstring>
#include<string>

int main()
{
    cv::Mat lena_img, gray_img;
    lena_img = cv::imread("D:\\MyFiles\\Year1a\\cv\\week2_edges\\lena_std.png");
    cv::cvtColor(lena_img, gray_img, cv::COLOR_BGR2GRAY);
    if(gray_img.empty())
    {
        std::cout << "Cannot load image!!!" << std::endl;
        return -1;
    }

    int lena_height = gray_img.rows; // 竖边长度
    int lena_width = gray_img.cols; // 横边长度
    int lena_channels = gray_img.channels(); // 读成灰度值，单通道，1

    // 图像中的每个点计算CRF
    // 遍历每个像素点，计算八个值，取min 
    // 滑窗大小无所谓，我只要实现对每行的八个方向的减法，再按照滑窗大小相加比大小即可
    int slid_win_size = 2;
    int half_size = 0;
    while(slid_win_size % 2 == 0)
    {
        std::cout << "Enter an ODD number as the size of the sliding window for Moravec algorithm: ";
        std::cin >> slid_win_size;
    }
    
    // 八个方向的差
    std::vector<cv::Mat> RowDiffMaps(8, cv::Mat(lena_height, lena_width, CV_64FC1, cv::Scalar(0)));
    for(int row = 0; row < 8; row++)
    {
        int di, dj;
        di = (row < 3) ? -slid_win_size : ((row > 4) ? slid_win_size : 0);
        if(row == 0 || row == 3 || row == 5)
            dj = -slid_win_size;
        else if(row == 1 || row == 6)
            dj = 0;
        else if(row == 2 || row == 4 || row == 7)
            dj = slid_win_size;
        for(int i = 0; i < lena_height; i++)
        {
            double * RowPtr = RowDiffMaps[row].ptr<double>(i);
            uchar * ImgPtr = gray_img.ptr<uchar>(i);
            uchar * ImgPtr_prev = gray_img.ptr<uchar>(std::min(lena_height - 1, std::max(0, i + di)));
            for(int j = 0; j < lena_width; j++)
            {
                RowPtr[j] = ImgPtr[j] - ImgPtr_prev[std::min(lena_width - 1, std::max(0, j + dj))];
            }
        }
        // cv::imwrite("D:\\MyFiles\\Year1a\\cv\\week2_edges\\lena_diff_" + std::to_string(row) + ".jpg", RowDiffMaps[row]);
    }

    // 按照窗口大小相加
    // Map(i, j)的值，是以(i, j)为中心的滑窗范围内所有差值平方相加，然后再各个通道比大小
    // 遍历每个(i, j)，计算八个方向的V值，并取最小送入CRFMap
    cv::Mat CRFMap(lena_height, lena_width, CV_64FC1, cv::Scalar(0));
    cv::Mat RowSumMap(slid_win_size, lena_width, CV_64FC1, cv::Scalar(0));
    cv::Mat FullSumMap(lena_height, lena_width, CV_64FC1, cv::Scalar(0)); // 即CRFMap
    for(int row = 0; row < 8; row++)
    {
        for(int i = 0; i < lena_height; i++)
        {
            double * RowSPtr = RowSumMap.ptr<double>(i % slid_win_size);
            double * RDPtr = RowDiffMaps[row].ptr<double>(i);
            // 行相加
            for(int j = 0; j < lena_width; j++)
            {
                RowSPtr[j] = 0;
                // 这里和之前写的代码不同，越界的值应记为0，而非以往的边界padding
                for(int dx = -half_size; dx <= half_size; dx++)
                {
                    if(dx + j < 0 || dx + j > lena_width - 1)
                    {
                        continue;
                    }
                    RowSPtr[j] += RDPtr[j + dx] * RDPtr[j + dx];
                }
            }
            // 列相加
            if(i > half_size)
            {
                int idx_row_calc = i - half_size;
                double * FSPtr = FullSumMap.ptr<double>(idx_row_calc);
                for(int dx = -half_size; dx <= half_size; dx++)
                {
                    if(idx_row_calc + dx < 0 || idx_row_calc + dx > lena_height - 1)
                    {
                        continue;
                    }
                    int idx_rsum_row = (idx_row_calc + dx) % slid_win_size;
                    double * RSMPtr = RowSumMap.ptr<double>(idx_rsum_row);
                    for(int j = 0; j < lena_width; j++)
                    {
                        FSPtr[j] += RSMPtr[j];
                    }
                }
            }
        }
        for(int i = lena_height - half_size; i <= lena_height - 1; i++)
        {
            int idx_row_calc = i;
            double * FSPtr = FullSumMap.ptr<double>(idx_row_calc);
            for(int dx = -half_size; dx <= half_size; dx++)
            {
                if(idx_row_calc + dx < 0 || idx_row_calc + dx > lena_height - 1)
                {
                    continue;
                }
                int idx_rsum_row = (idx_row_calc + dx) % slid_win_size;
                double * RSMPtr = RowSumMap.ptr<double>(idx_rsum_row);
                for(int j = 0; j < lena_width; j++)
                {
                    FSPtr[j] += RSMPtr[j];
                }
            }
        }
        if(row == 0)
        {
            CRFMap = FullSumMap;
        }
        else
        {
            // cv::Mat 逐值取小即可
            cv::min(CRFMap, FullSumMap, CRFMap);
        }
    }
    // cv::imwrite("D:\\MyFiles\\Year1a\\cv\\week2_edges\\lena_crf.jpg", CRFMap);

    // thresholding
    cv::Mat CRFFlat = CRFMap.reshape(0, 1);
    std::vector<cv::Point> crf_points;
    cv::findNonZero(CRFMap, crf_points);
    int crf_points_num = crf_points.size();
    int crf_points_threshed_num = static_cast<double>(crf_points_num * (1 - 0.99) - 1); // 可取其他比率
    std::nth_element(
        crf_points.begin(),
        crf_points.begin() + crf_points_threshed_num,
        crf_points.end(),
        [&] (cv::Point& a, cv::Point& b)
        {
            return (CRFMap.at<double>(a) > CRFMap.at<double>(b)); // 降序
        }
    );
    // crf_points从begin()到crf_points_num * (1 - 0.99) - 1的点即为threshold之后的值

    // nms，这里仍然采用anms
    struct PointRadii{
        cv::Point point;
        double radius;
    };
    std::vector<PointRadii> points_and_radii;
    for(int i = 0; i <= crf_points_threshed_num; i++)
    {
        PointRadii point_rad;
        point_rad.point = crf_points[i];
        double crf_val = CRFMap.at<double>(point_rad.point);
        double rad_min = 1.79769e+30; // inf 1.79769e+308
        double c_ratio = 0.9; // 可以取其他比率
        for(int j = 0; j <= crf_points_threshed_num; j++)
        {
            if(i == j)
            {
                continue;
            }
            double crf_val_j = CRFMap.at<double>(crf_points[j]);
            if(crf_val_j - c_ratio * crf_val > 1e-6)
            {
                double rad_now = norm(point_rad.point - crf_points[j]);
                rad_min = std::min(rad_min, rad_now);
            }
        }
        point_rad.radius = rad_min;
        points_and_radii.push_back(point_rad);
    }
    int anms_thresh = static_cast<int>(crf_points_threshed_num * 0.05); // 比率可以改
    std::nth_element(
        points_and_radii.begin(),
        points_and_radii.begin() + anms_thresh,
        points_and_radii.end(),
        [&] (PointRadii& a, PointRadii& b)
        {
            return (a.radius > b.radius); // 降序
        }
    );
    for(int i = 0; i <= anms_thresh; i++)
    {
        cv::circle(lena_img, points_and_radii[i].point, 5, cv::Scalar(0, 0, 255), 2);
    }
    cv::imwrite("D:\\MyFiles\\Year1a\\cv\\week2_edges\\lena_circled_moravec.jpg", lena_img);

    return 0;
}