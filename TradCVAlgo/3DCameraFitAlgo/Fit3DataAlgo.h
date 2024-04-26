#pragma once
#ifndef FIT3DATAALGO
#define FIT3DATAALGO
#include <chrono>
#include "pcl/point_cloud.h"
#include <pcl/visualization/pcl_visualizer.h>


struct QuadraticModel3D
{
	//曲面参数
	double a = 0, b = 0, c = 0, d = 0, e = 0, f = 0;
};

struct PlaneFormula
{
	//平面参数
	double a, b, c, d;
};

class Fit3DataAlgo
{
public:
	Fit3DataAlgo();
	~Fit3DataAlgo();
	int run(cv::Mat& input_image)
private:
	void RandomSample(cv::Mat& input_height,
		std::vector<cv::Point3d>& output_pts, int sample_num);
	QuadraticModel3D fitQuadraticRANSAC(const std::vector<cv::Point3d>& points, int maxIterations,
		double distanceThreshold, int minInliers);
	void RandomUniformitySample(cv::Mat& input_height,
		std::vector<cv::Point3d>& output_pts);
	PlaneFormula FitPlaneRANSAC(const std::vector<cv::Point3d>& points, int maxIterations,
		double distanceThreshold);
};
#endif
