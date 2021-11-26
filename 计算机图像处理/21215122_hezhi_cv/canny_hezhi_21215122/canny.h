#ifndef CANNY_H_
#define CANNY_H_

#include "CImg.h"
#include <string>

using namespace std;
using namespace cimg_library;

class Canny 
{
private:
	CImg<int> img;
	int rows;
	int cols;
	int *smoothedim;
	int *dx;
	int *dy;
	float *ddirection;//float *dirim;  //梯度的方向
	int *magnitude; //梯度的幅值
	int *nms;   //非极大值抑制后得到矩阵
	int *edge;  //边缘数组

public:
	Canny();
	Canny(string name, string format);
	~Canny();

	// 将图像变成灰度图
	void rgbToGray();
	CImg<int> rgbToGrayTest();

	// 进行高斯滤波
	void gaussian_smooth(float sigma);
	CImg<int> gaussian_smooth_test(float sigma);

	// 计算x,y方向的一阶导数
	void derrivative_x_y();

	// 计算梯度向上的方向，以正x轴为逆时针方向指定的弧度
	void radian_direction(int xdirtag, int ydirtag);

	// 计算梯度的幅值
	void magnitude_x_y();
	CImg<int> magnitude_x_y_test();

	// 进行非极大值抑制
	void non_max_supp();
	CImg<int> non_max_supp_test();

	// 进行双阈值检测
	void apply_hysteresis(float tlow, float thigh);
	CImg<int> apply_hysteresis_test(float tlow, float thigh);

	// 默认参数设置高斯过滤标准差为2.0，低阈值为0.25，高阈值为0.75
	CImg<int> canny_image();

	// 整合所有获取最后的边缘图, sigma表示高斯过滤的参数，tlow和thigh为两个阈值
	CImg<int> canny_image(int sigma, float tlow, float thigh);

	// 选出两个边缘点较近的距离连线
	CImg<int> canny_line(CImg<int> picture, int distance);

	// 删掉长度小于20的边缘线
	CImg<int> delete_line(CImg<int> picture, int distance);

	// 显示图像
	void display();

	// 保存图像
	void save(char *path);
};

#endif