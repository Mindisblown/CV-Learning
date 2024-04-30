/*
直线拟合
入参:
	input_points - 需要拟合的所有点数
	line_params - 最终拟合得到的直线
	dis_thre - 距离阈值
	max_iter - 拟合次数
	init_num - 初始拟合点的数量
*/
void FitLineRansac(std::vector<cv::Point2f>& input_points, cv::Vec4d& line_params,
	float dis_thre, int max_iter, int init_num)
{
	std::set<int> random_index; // 随机点的索引
	std::vector<cv::Point2f> random_points; // 随机点集合
	std::vector<cv::Point2f> temp_inliners; // 内集点
	int points_size = input_points.size();
	if (points_size < 2)
	{
		return;
	}
	int best_scores = 0; // 用于记录最大内点

	cv::RNG rng_generate;
	// 首先循环迭代次数
	for (int i = 0; i < max_iter; ++i)
	{
		random_index.clear();
		random_points.clear();
		temp_inliners.clear();
		// 选取随机点
		if (init_num == 2)
		{
			int p1 = 0, p2 = 0;
			while (p1 == p2)
			{
				p1 = rng_generate(points_size);
				p2 = rng_generate(points_size);
			}
			random_index.insert(p1);
			random_index.insert(p2);
			random_points.emplace_back(input_points[p1]);
			random_points.emplace_back(input_points[p2]);
		}
		else
		{
			for (int j = 0; j < init_num; ++j)
			{
				int p1 = rng_generate(points_size);
				random_index.insert(p1);
				random_points.emplace_back(input_points[p1]);
			}

		}
		// 使用随机点拟合直线
		cv::Vec4d temp_line;
		cv::fitLine(random_points, temp_line, cv::DIST_L2, 0, 0.01, 0.01);
		double temp_k = temp_line[1] / temp_line[0]; // 斜率k
		double temp_b = temp_line[3] - temp_k * temp_line[2]; // 截距b

		int temp_scores = 0;
		// 计算余下测试点是否在直线上
		for (int pt_index = 0; pt_index < points_size; ++pt_index)
		{
			if (random_index.find(pt_index) == random_index.end())
			{
				// 水平直线
				double dis_error = temp_k * input_points[pt_index].x + temp_b - input_points[pt_index].y;
				
				// 垂直直线
				// double dis_error = ((input_points[pt_index].y - temp_b) / temp_k) - input_points[pt_index].x;
				
				dis_error *= dis_error; // 平方项为误差
				if (dis_error < dis_thre)
				{
					temp_scores += 1;
					temp_inliners.emplace_back(input_points[pt_index]);
				}
			}
		}

		if (temp_scores > best_scores)
		{
			cv::fitLine(temp_inliners, line_params, cv::DIST_L2, 0, 0.01, 0.01);
			best_scores = temp_scores;
		}

	}

}




const float One = 1.0, Two = 2.0, Three = 3.0, Four = 4.0, Five = 5.0, Six = 6.0, Ten = 10.0;

// 圆拟合边缘点数据类型
class PointsData
{
public:
	int n;
	std::vector<float> X;
	std::vector<float> Y;
	float meanX, meanY;

public:
	PointsData();
	PointsData(int N, float dataXY[], int step);
	~PointsData();

public:
	void means();
};

// 拟合圆数据类型
class CircleData
{
public:
	float a, b, r, s, g, Gx, Gy;
	int i, j;

public:
	CircleData();
	~CircleData();
};

// * * * * * * * * * * * * * Points Class * * * * * * * * * * * * *
PointsData::PointsData()
{
	n = 0;
	X.resize(n);
	Y.resize(n);
	for (int i = 0; i < n; ++i)
	{
		X[i] = 0.;
		Y[i] = 0.;
	}
}

PointsData::PointsData(int N, float dataXY[], int step)
{
	n = N / step;
	X.resize(n);
	Y.resize(n);
	for (int i = 0; i < n; i++)
	{
		X[i] = (float)dataXY[2 * i * step];
		Y[i] = (float)dataXY[2 * i * step + 1];
	}
}

void PointsData::means()
{
	meanX = 0.;
	meanY = 0.;

	for (int i = 0; i < n; i++)
	{
		meanX += X[i];
		meanY += Y[i];
	}
	meanX /= n;
	meanY /= n;
}

PointsData::~PointsData() {}

// * * * * * * * * * * * * * Circle Class * * * * * * * * * * * * *
CircleData::CircleData()
{
	a = 0.; b = 0.; r = 1.; s = 0.; i = 0; j = 0;
}

CircleData::~CircleData() {}

void CalcDist2D(PointsData& data, CircleData& circle, std::vector<float>& dist)
{
	float dx, dy, sum_dist = 0;
	for (int i = 0; i < data.n; i++)
	{
		dx = data.X[i] - circle.a;
		dy = data.Y[i] - circle.b;
		dist[i] = fabs(sqrt(dx * dx + dy * dy) - circle.r);
		sum_dist += dist[i];
	}
}

void FitCircleByTaubin(PointsData& pt_data, CircleData& c_data)
{
	int i, iter, IterMAX = 99;

	float Xi, Yi, Zi;
	float Mz, Mxy, Mxx, Myy, Mxz, Myz, Mzz, Cov_xy, Var_z;
	float A0, A1, A2, A22, A3, A33;
	float Dy, xnew, x, ynew, y;
	float DET, Xcenter, Ycenter;
	pt_data.means();

	Mxx = Myy = Mxy = Mxz = Myz = Mzz = 0.;

	for (int i = 0; i < pt_data.n; ++i)
	{
		Xi = float(pt_data.X[i]) - pt_data.meanX;   //  centered x-coordinates
		Yi = float(pt_data.Y[i]) - pt_data.meanY;   //  centered y-coordinates
		Zi = Xi * Xi + Yi * Yi;

		Mxy += Xi * Yi;
		Mxx += Xi * Xi;
		Myy += Yi * Yi;
		Mxz += Xi * Zi;
		Myz += Yi * Zi;
		Mzz += Zi * Zi;
	}
	Mxx /= pt_data.n;
	Myy /= pt_data.n;
	Mxy /= pt_data.n;
	Mxz /= pt_data.n;
	Myz /= pt_data.n;
	Mzz /= pt_data.n;

	Mz = Mxx + Myy;
	Cov_xy = Mxx * Myy - Mxy * Mxy;
	Var_z = Mzz - Mz * Mz;
	A3 = Four * Mz;
	A2 = -Three * Mz*Mz - Mzz;
	A1 = Var_z * Mz + Four * Cov_xy*Mz - Mxz * Mxz - Myz * Myz;
	A0 = Mxz * (Mxz*Myy - Myz * Mxy) + Myz * (Myz*Mxx - Mxz * Mxy) - Var_z * Cov_xy;
	A22 = A2 + A2;
	A33 = A3 + A3 + A3;

	for (x = 0., y = A0, iter = 0; iter < IterMAX; iter++)  // usually, 4-6 iterations are enough
	{
		Dy = A1 + x * (A22 + A33 * x);
		xnew = x - y / Dy;
		if ((xnew == x) || (!isfinite(xnew))) break;
		ynew = A0 + xnew * (A1 + xnew * (A2 + xnew * A3));
		if (abs(ynew) >= abs(y))  break;
		x = xnew;  y = ynew;
	}

	DET = x * x - x * Mz + Cov_xy;
	Xcenter = (Mxz*(Myy - x) - Myz * Mxy) / DET / Two;
	Ycenter = (Myz*(Mxx - x) - Mxz * Mxy) / DET / Two;

	c_data.a = Xcenter + pt_data.meanX;
	c_data.b = Ycenter + pt_data.meanY;
	c_data.r = sqrt(Xcenter*Xcenter + Ycenter * Ycenter + Mz);
	c_data.s = Sigma(pt_data, c_data);
	c_data.i = 0;
	c_data.j = iter;  //  return the number of iterations, too
}

void GetInlier(PointsData& pt_data, PointsData& in_pt_data, CircleData& c_data, int* inlier, float ratio)
{
	int count = pt_data.n, k = count * ratio;
	std::vector<float> dist(count), tmp;
	CalcDist2D(pt_data, c_data, dist);
	tmp = dist;		nth_element(tmp.begin(), tmp.begin() + k, tmp.end());
	std::vector<cv::Point2f> ps;
	float th = tmp[k];
	for (int i = 0; i < count; i++) {
		inlier[i] = (dist[i] < th) ? 1 : 0;
		if (inlier[i])	ps.push_back(cv::Point(pt_data.X[i], pt_data.Y[i]));
	}
	in_pt_data = PointsData(ps.size(), (float*)&ps[0], 1);
}

/*
圆拟合
入参:
	pt_data - 需要拟合的所有点数
	c_data - 最终拟合得到的圆
说明:
	当边缘点每次获取都很稳定无干扰和直接拟合一次，否则
	开启多次拟合，耗时增大不到100ms
*/

void FitCircle(PointsData& pt_data, CircleData& c_data)
{
	// 拟合效果不好可多次拟合
	std::vector<int> inlier(pt_data.n);
	PointsData in_pt_data;
	for (int i = 0; i < 2; ++i)
	{
		FitCircleByTaubin(pt_data, c_data);
		GetInlier(pt_data, in_pt_data, c_data, &inlier[0], 0.80);
		pt_data = in_pt_data;
	}

	FitCircleByTaubin(pt_data, c_data);
}

