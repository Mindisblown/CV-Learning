
int run(cv::Mat& input_data)
{
    cv::Mat loc_binary_image; //模型推理结果
    cv::Mat non_zeros_pts;
    cv::findNonZero(loc_binary_image, non_zeros_pts);
    cv::Mat foreground_img(non_zeros_pts.rows, 1, algo_input_image_.type());
    for(int i = 0; i < non_zeros_pts.rows; ++i)
    {
        //不对所有点计算，只针对前景点计算
        cv::Point pt = non_zeros_pts.at<cv::Point>(i);
        pt.x = pt.x * algo_input_image_.cols / loc_binary_image.cols;
        pt.y = pt.y * algo_input_image_.rows / loc_binary_image.rows;
        foreground_img.at<cv::Vec3b>(i) = algo_input_image_.at<cv::Vec3b>(pt);
    }
    cv::cvtColor(foreground_img, foreground_img, cv::COLOR_RGB2HSV);
    std::vector<cv::Mat> hsv_channel;
    cv::split(foreground_img, hsv_channel);
    cv::Mat s_space_img = hsv_channel[1];
    cv::Mat v_space_img = hsv_channel[2];
    int hist_size = 256;
    float range[] = {20, 200};
    const float* hist_range = {range};
    cv::Mat s_hist, v_hist;

    cv::calcHist(&s_space_img, 1, 0, cv::Mat(), s_hist, 1, &hist_size, &hist_range);
    cv::calcHist(&v_space_img, 1, 0, cv::Mat(), v_hist, 1, &hist_size, &hist_range);
    double s_max_value, v_max_value;
    cv::Point s_max_pt, v_max_pt;
    cv::minMaxLoc(s_hist, NULL, &s_max_value, NULL, &s_max_pt);
    cv::minMaxLoc(v_hist, NULL, &v_max_value, NULL, &v_max_pt);

    int s_space_peak = s_max_pt.y, v_space_peak = v_max_pt.y;
}
