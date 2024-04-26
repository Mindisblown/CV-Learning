#pragma execution_character_set("utf-8")
#include "Fit3DataAlgo.h"
#include <iostream>
#include <string>
#include <memory>
#include <random>
#include <utility>
#include <numeric>


void normalizeImage(cv::Mat &mat)
{
	mat.convertTo(mat, CV_64F);
	double v_max = -1;
	double v_min = 99999;
	for (int row = 0; row < mat.rows; row++)
	{
		double *curr_depth = mat.ptr<double>(row);
		for (int col = 0; col < mat.cols; col++)
		{
			double z = curr_depth[col];
			if (z > 0)
			{
				if (z > v_max)
					v_max = z;
				if (z < v_min)
					v_min = z;
			}
		}
	}

	for (int row = 0; row < mat.rows; row++)
	{
		auto *curr_depth = mat.ptr<double>(row);
		for (int col = 0; col < mat.cols; col++)
		{
			auto z = curr_depth[col];
			if (z > 0)
			{
				auto cc = (z - v_min) / (v_max - v_min);
				curr_depth[col] = cc * 255;
			}
		}
	}
	mat.convertTo(mat, CV_8UC1);
	cv::cvtColor(mat, mat, cv::COLOR_GRAY2RGB);
}

// 点到平面的距离
double Point2PlaneDistance(const PlaneFormula& plane, const cv::Point3d& point)
{
	double x = point.x;
	double y = point.y;
	double z = point.z;
	// ax+by+cz+d=0
	return std::abs(plane.a * point.x + plane.b * point.y + plane.c * point.z + plane.d)
		/ std::sqrt(plane.a * plane.a + plane.b * plane.b + plane.c * plane.c);
}

// 求两个平面直接的距离
double pointToPlaneDistance(double x, double y, double z, double a, double b, double c, double d) {
	return std::abs(a*x + b * y + c * z + d);
}

// 点到曲面的距离
double computeDistance(const QuadraticModel3D& model, const cv::Point3d& point)
{
	double x = point.x;
	double y = point.y;
	double z = point.z;
	// z=ax^2+by^2+cxy+dx+ey+f
	return std::abs(z - (model.a * x * x +
		model.b * y * y + model.c * x * y + model.d * x + model.e * y + model.f));
}



Fit3DataAlgo::Fit3DataAlgo() {}

Fit3DataAlgo::~Fit3DataAlgo() {}


void Fit3DataAlgo::RandomSample(cv::Mat& input_height,
	std::vector<cv::Point3d>& output_pts, int sample_num)
{
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_int_distribution<int> rowDist(0, input_height.rows - 3);
	std::uniform_int_distribution<int> colDist(0, input_height.cols - 3);
	
	for (int i = 0; i < sample_num; i++)
	{
		int y = rowDist(gen);
		int x = colDist(gen);
		cv::Point temp_pt(x, y);
		
		double* height_z = input_height.ptr<double>(y);
		double z = height_z[x];
		if (z == 0)
		{
			continue;
		}
		output_pts.emplace_back(double(x), double(y), z);
	}
}

QuadraticModel3D Fit3DataAlgo::fitQuadraticRANSAC(const std::vector<cv::Point3d>& points,
	int maxIterations, double distanceThreshold, int minInliers)
{
	int numPoints = points.size();
	

	int bestInlierCount = 0;
	QuadraticModel3D bestModel;

	cv::RNG rng;

	for (int iteration = 0; iteration < maxIterations; ++iteration)
	{
		// 随机选择5个点作为内点集合
		std::vector<cv::Point3d> inliers;
		for (int i = 0; i < 6; ++i)
		{
			int randomIndex = rng.uniform(0, numPoints);
			inliers.push_back(points[randomIndex]);
		}

		// 使用内点集合拟合二次函数模型
		cv::Mat A(inliers.size(), 6, CV_64F);
		cv::Mat B(inliers.size(), 1, CV_64F);
		for (int i = 0; i < inliers.size(); ++i)
		{
			double x = inliers[i].x;
			double y = inliers[i].y;
			double z = inliers[i].z;
			A.at<double>(i, 0) = x * x;
			A.at<double>(i, 1) = y * y;
			A.at<double>(i, 2) = x * y;
			A.at<double>(i, 3) = x;
			A.at<double>(i, 4) = y;
			A.at<double>(i, 5) = 1.0;
			B.at<double>(i, 0) = z;
		}
		// z=ax^2+by^2+cxy+dx+ey+f
		cv::Mat X;
		cv::solve(A, B, X, cv::DECOMP_NORMAL);

		QuadraticModel3D model;
		model.a = X.at<double>(0, 0);
		model.b = X.at<double>(1, 0);
		model.c = X.at<double>(2, 0);
		model.d = X.at<double>(3, 0);
		model.e = X.at<double>(4, 0);

		// 计算所有数据点到拟合的二次函数模型的距离
		int inlierCount = 0;
		for (int i = 0; i < numPoints; ++i)
		{
			double distance = computeDistance(model, points[i]);
			if (distance < distanceThreshold)
			{
				inlierCount++;
			}
		}

		// 更新最优模型
		if (inlierCount > bestInlierCount)
		{
			bestInlierCount = inlierCount;
			bestModel = model;

			//// 如果内点数量已经满足要求，提前结束迭代
			//if (inlierCount >= minInliers)
			//{
			//	break;
			//}
		}
	}

	//// 使用所有内点重新拟合二次函数模型
	//cv::Mat A(bestInlierCount, 3, CV_64F);
	//cv::Mat B(bestInlierCount, 1, CV_64F);
	//int index = 0;
	//for (int i = 0; i < numPoints; ++i)
	//{
	//	double distance = computeDistance(bestModel, points[i]);
	//	if (distance < distanceThreshold)
	//	{
	//		double x = points[i].x;
	//		double y = points[i].y;
	//		double z = points[i].z;
	//		A.at<double>(index, 0) = x * x;
	//		A.at<double>(index, 1) = x;
	//		A.at<double>(index, 2) = 1.0;
	//		B.at<double>(index, 0) = y;
	//		index++;
	//	}
	//}
	//cv::Mat X;
	//cv::solve(A, B, X, cv::DECOMP_SVD);
	//QuadraticModel3D finalModel;
	//finalModel.a = X.at<double>(0, 0);
	//finalModel.b = X.at<double>(1, 0);
	//finalModel.c = X.at<double>(2, 0);
	//if (abs(bestModel.a) < 5e-06)
	//{
	//	std::cout << "plane" << std::endl;
	//}

	//cv::Mat A = (cv::Mat_<double>(2, 2) << 2 * bestModel.a, bestModel.c, bestModel.b, 2 * bestModel.b);
	//cv::Mat B = (cv::Mat_<double>(2, 1) << -bestModel.d, -bestModel.e);

	//// 解方程组
	//cv::Mat solution;
	//cv::solve(A, B, solution, cv::DECOMP_NORMAL);

	//// 获取极值点的坐标
	//double x0 = solution.at<double>(0, 0);
	//double y0 = solution.at<double>(1, 0);

	//// 计算极值点的 z 值
	//double z0 = bestModel.a * x0 * x0 + bestModel.b * y0 * y0 +
	//	bestModel.c * x0 * y0 + bestModel.d * x0 + bestModel.e * y0 + bestModel.f;
	//std::cout << "peaking value" << z0 << std::endl;
	return bestModel;
}


void Fit3DataAlgo::RandomUniformitySample(cv::Mat& input_height,
	std::vector<cv::Point3d>& output_pts)
{
	std::vector<cv::Point> non_zeros_idx;
	cv::findNonZero(input_height, non_zeros_idx);
	for (int i = 0; i < non_zeros_idx.size(); i += 10)
	{
		int x = non_zeros_idx[i].x;
		int y = non_zeros_idx[i].y;
		double* height_z = input_height.ptr<double>(y);
		double z = height_z[x];
		output_pts.emplace_back(double(x), double(y), z);
	}
}

PlaneFormula Fit3DataAlgo::FitPlaneRANSAC(const std::vector<cv::Point3d>& points,
	int maxIterations, double distanceThreshold)
{

	int numPoints = points.size();
	
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_int_distribution<> dist(0, points.size() - 1);

	PlaneFormula bestPlane;
	int bestInliers = 0;

	for (int i = 0; i < maxIterations; ++i) {
		// 随机选择三个点作为平面模型的一组参数
		int index1 = dist(gen);
		int index2 = dist(gen);
		int index3 = dist(gen);

		const cv::Point3d& p1 = points[index1];
		const cv::Point3d& p2 = points[index2];
		const cv::Point3d& p3 = points[index3];

		/*double v1x = p2.x - p1.x;
		double v1y = p2.y - p1.y;
		double v1z = p2.z - p1.z;

		double v2x = p3.x - p1.x;
		double v2y = p3.y - p1.y;
		double v2z = p3.z - p1.z;*/

		//// 计算法向量
		//PlaneFormula model;
		//model.a = (v1y * v2z) - (v1z * v2y);
		//model.b = (v1z * v2x) - (v1x * v2z);
		//model.c = (v1x * v2y) - (v1y * v2x);
		//model.d = -(model.a * p1.x + model.b * p1.y + model.c * p1.z);

		// 计算平面模型的参数(ax + by + cz + d = 0)
		double nx = (p2.y - p1.y) * (p3.z - p1.z) - (p2.z - p1.z) * (p3.y - p1.y);
		double ny = (p2.z - p1.z) * (p3.x - p1.x) - (p2.x - p1.x) * (p3.z - p1.z);
		double nz = (p2.x - p1.x) * (p3.y - p1.y) - (p2.y - p1.y) * (p3.x - p1.x);

		double length = std::sqrt(nx * nx + ny * ny + nz * nz);
		nx /= length;
		ny /= length;
		nz /= length;

		// 计算平面的参数
		double d = -(nx * p1.x + ny * p1.y + nz * p1.z);

		//double d = -(a * p1.x + b * p1.y + c * p1.z);

		PlaneFormula model;
		model.a = nx;
		model.b = ny;
		model.c = nz;
		model.d = d;

		// 计算拟合误差小于阈值的内点数量
		int inliers = 0;
		for (const auto& point : points) {
			double error = Point2PlaneDistance(model, point);
			if (error < distanceThreshold) {
				++inliers;
			}
		}

		// 更新参数和内点数量
		if (inliers > bestInliers) {
			bestInliers = inliers;
			bestPlane = model;
		}
	}
	//std::cout << "best pts " << bestInliers << std::endl;
	return bestPlane;
}


int Fit3DataAlgo::run(cv::Mat& input_image)
{
	cv::Mat crop_height;
	double z_axis_resolution = 0.0003;
	cv::medianBlur(input_image, crop_height, 5);
	crop_height.convertTo(crop_height, CV_64F);
	crop_height = crop_height * z_axis_resolution; // 转换实际距离

	
	//// test
	//pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_ave(new pcl::PointCloud<pcl::PointXYZ>);
	////pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_vex(new pcl::PointCloud<pcl::PointXYZ>);
	//pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> handle_ave(cloud_ave, 0, 255, 0);
	////pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> handle_vex(cloud_vex, 255, 0, 0);
	//std::vector<cv::Point> non_zeros_idx;
	//cv::findNonZero(crop_height, non_zeros_idx);
	//for (int i = 0; i < non_zeros_idx.size(); ++i)
	//{
	//	int x = non_zeros_idx[i].x;
	//	int y = non_zeros_idx[i].y;
	//	double* height_z = crop_height.ptr<double>(y);
	//	double z = height_z[x] * 50;
	//	cloud_ave->push_back(pcl::PointXYZ(x,
	//		y, z));
	//}
	////for (int i = 0; i < crop_height_pts.size(); ++i)
	////{
	////	// z=ax^2+by^2+cxy+dx+ey+f
	////	float x = crop_height_pts[i].x;
	////	float y = crop_height_pts[i].y;
	////	float z = model.a * x * x + model.b * y * y + model.c*x*y + model.d *x+model.e*y + model.f;
	////	cloud_vex->push_back(pcl::PointXYZ(x, y, z * 100));
	////	
	////}
	//pcl::visualization::PCLVisualizer viewer("3D Viewer");
	//viewer.setBackgroundColor(0, 0, 0);
	//viewer.addPointCloud<pcl::PointXYZ>(cloud_ave, handle_ave, "cloud_ave");
	////viewer.addPointCloud<pcl::PointXYZ>(cloud_vex, handle_vex, "cloud_vex");
	//viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "cloud_ave");
	////viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "cloud_vex");
	//// 设置可视化器参数
	//viewer.setRepresentationToWireframeForAllActors();
	//viewer.setShowFPS(false);
	//viewer.setSize(500, 500);
	//viewer.setCameraPosition(0, 0, 0, 0, 0, 0, 0);
	//// 显示点云和平面
	//while (!viewer.wasStopped()) {
	//	viewer.spinOnce();
	//}

	std::vector<cv::Point3d> crop_height_pts{};
	RandomUniformitySample(crop_height, crop_height_pts);
	int maxIterations = 500;
	double distanceThreshold = 0.5;
	int minInliers = 1000;

	if (crop_height_pts.size() < 5)
	{
		return -1;
	}
	//曲面模型
	QuadraticModel3D model = fitQuadraticRANSAC(crop_height_pts, maxIterations,
		distanceThreshold, minInliers);
	////pcl测试凹凸面
	//pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_ave(new pcl::PointCloud<pcl::PointXYZ>);
	//pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_vex(new pcl::PointCloud<pcl::PointXYZ>);
	//pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> handle_ave(cloud_ave, 0, 255, 0);
	//pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> handle_vex(cloud_vex, 255, 0, 0);
	//for (int i = 0; i < crop_height_pts.size(); ++i)
	//{
	//	cloud_ave->push_back(pcl::PointXYZ(crop_height_pts[i].x,
	//		crop_height_pts[i].y, crop_height_pts[i].z * 20));
	//}
	//for (int i = 0; i < crop_height_pts.size(); ++i)
	//{
	//	// z=ax^2+by^2+cxy+dx+ey+f
	//	float x = crop_height_pts[i].x;
	//	float y = crop_height_pts[i].y;
	//	float z = model.a * x * x + model.b * y * y + model.c*x*y + model.d *x+model.e*y + model.f;
	//	cloud_vex->push_back(pcl::PointXYZ(x, y, z * 20));
	//	
	//}
	//pcl::visualization::PCLVisualizer viewer("3D Viewer");
	//viewer.setBackgroundColor(0, 0, 0);
	//viewer.addPointCloud<pcl::PointXYZ>(cloud_ave, handle_ave, "cloud_ave");
	//viewer.addPointCloud<pcl::PointXYZ>(cloud_vex, handle_vex, "cloud_vex");
	//viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "cloud_ave");
	//viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "cloud_vex");
	//// 设置可视化器参数
	//viewer.setRepresentationToWireframeForAllActors();
	//viewer.setShowFPS(false);
	//viewer.setSize(500, 500);
	//viewer.setCameraPosition(0, 0, 0, 0, 0, 0, 0);
	//// 显示点云和平面
	//while (!viewer.wasStopped()) {
	//	viewer.spinOnce();
	//}
	if (crop_height_pts.size() < 5)
	{
		return -1;
	}
	//平面模型
	PlaneFormula plane_model = FitPlaneRANSAC(crop_height_pts, 500, 0.3);
	////pcl测试平面
	//pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_ave(new pcl::PointCloud<pcl::PointXYZ>);
	//pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_vex(new pcl::PointCloud<pcl::PointXYZ>);
	//pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> handle_ave(cloud_ave, 0, 255, 0);
	//pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> handle_vex(cloud_vex, 255, 0, 0);
	//for (int i = 0; i < crop_height_pts.size(); ++i)
	//{
	//	cloud_ave->push_back(pcl::PointXYZ(crop_height_pts[i].x,
	//		crop_height_pts[i].y, crop_height_pts[i].z * 20));
	//}
	//for (int i = 0; i < crop_height_pts.size(); ++i)
	//{
	//	// ax+by+cz+d=0
	//	float x = crop_height_pts[i].x;
	//	float y = crop_height_pts[i].y;
	//	float z = (-plane_model.a * x - plane_model.b * y - plane_model.d) / plane_model.c;
	//	cloud_vex->push_back(pcl::PointXYZ(x, y, z * 20));
	//	
	//}
	//pcl::visualization::PCLVisualizer viewer("3D Viewer");
	//viewer.setBackgroundColor(0, 0, 0);
	//viewer.addPointCloud<pcl::PointXYZ>(cloud_ave, handle_ave, "cloud_ave");
	//viewer.addPointCloud<pcl::PointXYZ>(cloud_vex, handle_vex, "cloud_vex");
	//viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "cloud_ave");
	//viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "cloud_vex");
	//// 设置可视化器参数
	//viewer.setRepresentationToWireframeForAllActors();
	//viewer.setShowFPS(false);
	//viewer.setSize(500, 500);
	//viewer.setCameraPosition(0, 0, 0, 0, 0, 0, 0);
	//// 显示点云和平面
	//while (!viewer.wasStopped()) {
	//	viewer.spinOnce();
	//}


	cv::Mat def_result_mask;
	// 找局部def位置
	cv::Mat sml_label, sml_stats, sml_cen;
	cv::Rect draw_rect;
	int nccomps1 = cv::connectedComponentsWithStats(def_result_mask,
		sml_label, sml_stats, sml_cen);
	int def_index = 0;
	// 转换用于绘制局部缺陷
	normalizeImage(algo_output_height_);
	for (int i = 1; i < nccomps1; ++i)
	{
		draw_rect.x = sml_stats.at<int>(i, 0);
		draw_rect.y = sml_stats.at<int>(i, 1);
		draw_rect.width = sml_stats.at<int>(i, 2);
		draw_rect.height = sml_stats.at<int>(i, 3);
		draw_rect = draw_rect & cv::Rect(0, 0, def_result_mask.cols, def_result_mask.rows);

		// 仅剩局部检测为缺陷的区域
		cv::Mat def_crop_img = crop_height(draw_rect).clone();
		// 对局部求最小，表示缺陷深度
		std::vector<cv::Point> non_zeros_idx;
		cv::findNonZero(def_crop_img, non_zeros_idx);
		double min_value = 100;
		cv::Point min_loc(0, 0);
		
		for (int j = 0; j < non_zeros_idx.size(); ++j)
		{
			int x = non_zeros_idx[j].x;
			int y = non_zeros_idx[j].y;
			double* height_z = def_crop_img.ptr<double>(y);
			double z = height_z[x];
			if (z < min_value)
			{
				min_value = z;
				/*min_loc.x = loc_box.x + draw_rect.x + x;
				min_loc.y = loc_box.y + draw_rect.y + y;*/
				min_loc.x =  draw_rect.x + x;
				min_loc.y =  draw_rect.y + y;
			}
		}
		
		double plane_distance = pointToPlaneDistance(min_loc.x, min_loc.y, min_value, plane_model.a,
			plane_model.b, plane_model.c, plane_model.d);
		char vis_plane_dis[15];
		sprintf_s(vis_plane_dis, sizeof(vis_plane_dis), "%.3f", plane_distance);
		if (plane_distance <= local_concave_thresh_)
		{
			
			continue;
		}
		
	}
	
	int acan_index = 0;
	// 输出拟合结果
	double plane_horizaontal = model.a * 1e5;
	

	/*normalizeImage(algo_input_height_);

	memset(DataBuffer, 0, Width Height * 3);
	memset(DataBuffer, 0, Width Height * 3);
	memcpy(DataBuffer, algo_input_height_.data,
		Width * Height * 3);
	memcpy(DataBuffer, algo_output_height_.data,
		Width * Height * 3);*/

	return 0;
}