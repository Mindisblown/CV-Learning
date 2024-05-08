float distance(cv::Point2f p1, cv::Point2f p2) {
	return std::sqrt((p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y));
}

bool FlatternCircle(const cv::Mat& inputImg, cv::OutputArray outputImg,
	const cv::Point& center, const int nRadius, const int nRingHeight, cv::Mat& mapx, cv::Mat& mapy)
{
	if (inputImg.empty())
		return false;

	//输出图像为矩形，高度为圆环高度，宽度为圆环外圆周长
	outputImg.create(cv::Size(nRadius * CV_2PI, nRingHeight), CV_8UC1);
	cv::Mat rectangle = outputImg.getMat();
	int rows = rectangle.rows;
	int cols = rectangle.cols;
	mapx = cv::Mat(rows, cols, CV_32FC1);	mapy = cv::Mat(rows, cols, CV_32FC1);
	for (int j = 0; j < rows; j++)
	{
		//	uchar* data = rectangle.ptr<uchar>(j);
		for (int i = 0; i < cols; i++)
		{
			//根据极坐标计算公式设置展平矩形对应位置的像素值
			double theta = CV_2PI / float(cols) * float(i + 1);
			double rho = nRadius - j - 1;
			int x = (float)rho * std::cos(theta) + 0.5;
			int y = (float)rho * std::sin(theta) + 0.5;
			mapx.at<float>(j, i) = x;	mapy.at<float>(j, i) = y;
			//	data[i] = inputImg.at<uchar>(y, x);
		}
	}
	cv::resize(mapx, mapx, cv::Size(1024, 30));
	cv::resize(mapy, mapy, cv::Size(1024, 30));
	cv::Mat mapx1 = mapx + center.x, mapy1 = mapy + center.y;

	cv::remap(inputImg, outputImg, mapx1, mapy1, CV_INTER_AREA);

	return true;
}

void makeTemplate(cv::Mat& src, cv::Mat& temp, cv::Mat& mapx, cv::Mat& mapy) {
	cv::Mat binary_img;
	cv::threshold(src, binary_img, 50, 255, 0);

	cv::Rect test_rect = cv::boundingRect(binary_img);
	//cv::rectangle(src, test_rect, cv::Scalar(255, 255, 255), 2);
	int w1 = src.cols, h = src.rows;
	// 中心点 圆环半径 圆环的宽度
	//FlatternCircle(src, temp, cv::Point(w1 / 2, h / 2), w1 / 3, w1 / 3 - 10, mapx, mapy);
	FlatternCircle(src, temp, cv::Point(test_rect.x + test_rect.width / 2, test_rect.y + test_rect.height / 2), 
		test_rect.width / 2, test_rect.width / 3 - 10, mapx, mapy);
	hconcat(temp, temp, temp);
}


int ProcessFrame(cv::Mat& srcFrame）
	
	cv::Mat gray_img = cv::imread("../../model-data/0-0_2023_11_14_14_38_58_938_284.png", 0);	
	//cv::resize(gray_img, gray_img, cv::Size(640, 640));
	cv::Mat tempalte_result = gray_img.clone(), map_x, map_y;
	int w = gray_img.cols, h = gray_img.rows;
	SimdGaussianBlur3x3(gray_img.data, w, w, h, 1, tempalte_result.data, w);
	makeTemplate(tempalte_result, gray_img, map_x, map_y);//极坐标展开制作模板
	tempalte_result = gray_img.clone();
	//cv::resize(gray_img, tempalte_result, cv::Size(gray_img.cols, 300));
	//cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();	clahe->setClipLimit(3);
	//clahe->apply(tempalte_result, tempalte_result);
	
	std::vector<cv::String> filenames;
	cv::String folder = "../../model-data/bottle-cover/qx-test/*.png";
	cv::glob(folder, filenames);
	for (size_t i = 0; i < filenames.size(); i++)
	{
		std::cout << filenames[i] << std::endl;
		cv::Mat img = cv::imread(filenames[i], 0);
		//cv::resize(img, img, cv::Size(640, 640));
		cv::resize(img, img, algo_input_image_.size());
		cv::Mat binary_img;
		cv::threshold(img, binary_img, 50, 255, 0);
		cv::Rect test_rect = cv::boundingRect(binary_img);
		//cv::rectangle(img, test_rect, cv::Scalar(255, 255, 255), 2);
		

		cv::Mat img_ori = cv::imread(filenames[i]);
		//cv::resize(img_ori, img_ori, cv::Size(640, 640));
		cv::Mat bw = img.clone(), tOut;
		cv::Point maxp1;	double p1;

		
		//int w1 = img.cols, h1 = img.rows;
		int w1 = test_rect.x + test_rect.width / 2, h1 = test_rect.y + test_rect.height / 2;
		//cv::Mat mapx1 = map_x + w1 / 2, mapy1 = map_y + h1 / 2;
		cv::Mat mapx1 = map_x + w1, mapy1 = map_y + h1;
		SimdGaussianBlur3x3(img.data, w1, w1, h1, 1, bw.data, w1);


		std::chrono::time_point<std::chrono::system_clock> start_time4 =
			std::chrono::system_clock::now();
		cv::remap(bw, img, mapx1, mapy1, CV_INTER_AREA);
		//resize(img, img, cv::Size(img.cols, 300));
		//clahe->apply(img, img);
		cv::matchTemplate(img, tempalte_result, tOut, 5);
		cv::minMaxLoc(tOut, NULL, &p1, NULL, &maxp1);
		//cv::Mat vis_img = tempalte_result.clone();
		//cv::cvtColor(vis_img, vis_img, cv::COLOR_GRAY2BGR);
		//cv::line(vis_img, cv::Point(maxp1.x, 0), cv::Point(maxp1.x, img.rows), cv::Scalar(0, 255, 0), 2);
		//cv::line(vis_img, cv::Point(maxp1.x + img.cols, 0), cv::Point(maxp1.x + img.cols, img.rows), cv::Scalar(0, 255, 0), 2);
		
		// 匹配的位置在整个图片宽度上的比例-用该比例来表示角度
		float angle = CV_2PI / float(img.cols) * float(maxp1.x+ 1);
		angle *= 180 / CV_PI;
		std::chrono::time_point<std::chrono::system_clock> end_time4 =
			std::chrono::system_clock::now();

		std::chrono::duration<double, std::milli> fp_ms4 = end_time4 - start_time4;

		std::cout << "infer time" << std::to_string(fp_ms4.count()) << std::endl;
	
		// Now rotate and scale fragment back, then find translation
		cv::Mat rot_mat = cv::getRotationMatrix2D(cv::Point(img_ori.cols / 2, img_ori.rows / 2), -angle, 1.0);

		// rotate and scale
		cv::Mat im1_rs;
		warpAffine(img_ori, im1_rs, rot_mat, img_ori.size());

	}
	
	return 0;
}
