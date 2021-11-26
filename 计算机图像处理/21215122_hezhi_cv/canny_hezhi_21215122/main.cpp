#include "canny.h"
#include <iostream>
#include <vector>

using namespace std;

void cannyStepByStep();  	 // 输出Canny算法每一步的图像
void cannyAllByOne();    	 // 通过Canny算法直接输出图像边缘
void deleteShortEdgeTest();  // 连相邻边并删短Edge测试
void sigmaTest();  			 // 高斯滤波调参测试
void hysteresisMaxValTest(); // 高阈值调参测试
void hysteresisMinValTest(); // 低阈值调参测试

int main() {
	// cannyStepByStep();
	// cannyAllByOne();	
	// deleteShortEdgeTest();
	// sigmaTest();
	// hysteresisMinValTest();
	hysteresisMaxValTest();

	return 0;
}

void cannyStepByStep()
{	
	string inputDir = "./test_Data";
	string outputDir = "./result_Data";
	string fileExt = "jpg";
	vector<string> filenameList = {"lena", "bigben", "stpietro", "3", "4", "20160326110137505"};

	for (vector<string>::iterator iter = filenameList.begin(); iter != filenameList.end(); iter++) {
		string fileName = *iter;
		string filePath = inputDir + "/" + fileName + "." + fileExt;

		Canny canny(filePath, fileExt);

		CImg<int> pic1 = canny.rgbToGrayTest();
		string filePath1 = outputDir + "/" + fileName + "_gray." + fileExt;
		char *path1 = (char *)filePath1.data();
		pic1.save(path1);

		CImg<int> pic2 = canny.gaussian_smooth_test(2.0);
		string filePath2 = outputDir + "/" + fileName + "_gaussian." + fileExt;
		char *path2 = (char *)filePath2.data();
		pic2.save(path2);

		CImg<int> pic3 = canny.magnitude_x_y_test();
		string filePath3 = outputDir + "/" + fileName + "_derrivative." + fileExt;
		char *path3 = (char *)filePath3.data();
		pic3.save(path3);

		CImg<int> pic4 = canny.non_max_supp_test();
		string filePath4 = outputDir + "/" + fileName + "_nms." + fileExt;
		char *path4 = (char *)filePath4.data();
		pic4.save(path4);

		CImg<int> pic5 = canny.apply_hysteresis_test(0.25, 0.75);
		string filePath5 = outputDir + "/" + fileName + "_hysteresis." + fileExt;
		char *path5 = (char *)filePath5.data();
		pic5.save(path5);
	}
}

void cannyAllByOne()
{
	string inputDir = "./test_Data";
	string outputDir = "./result_Data";
	string fileExt = "jpg";
	vector<string> filenameList = {"lena", "bigben", "stpietro", "3", "4", "20160326110137505"};

	for (vector<string>::iterator iter = filenameList.begin(); iter != filenameList.end(); iter++) {
		string filePath = inputDir + "/" + *iter + "." + fileExt;

		Canny img(filePath, fileExt);
		CImg<int> imgCanny = img.canny_image(2.0, 0.25, 0.75);

		string saveFilePath = outputDir + "/" + *iter + "_canny." + fileExt;
		char *savePath = (char *)saveFilePath.data();
		imgCanny.save(savePath);
	}
}

void deleteShortEdgeTest()
{
	Canny canny("./test_Data/lena.jpg", "jpg");
	CImg<int> imgCanny = canny.canny_image(2.0, 0.25, 0.75);
	CImg<int> img1 = canny.canny_line(imgCanny, 10);
	CImg<int> img2 = canny.delete_line(img1, 20);
	(imgCanny, img1, img2).display("Delete short edge!");
}

void sigmaTest()
{
	Canny canny1("./test_Data/lena.jpg", "jpg");
	CImg<int> img1 = canny1.canny_image(1.0, 0.25, 0.75);

	Canny canny2("./test_Data/lena.jpg", "jpg");
	CImg<int> img2 = canny2.canny_image(2.0, 0.25, 0.75);

	Canny canny3("./test_Data/lena.jpg", "jpg");
	CImg<int> img3 = canny3.canny_image(3.0, 0.25, 0.75);

	Canny canny4("./test_Data/lena.jpg", "jpg");
	CImg<int> img4 = canny4.canny_image(5.0, 0.25, 0.75);

	Canny canny5("./test_Data/lena.jpg", "jpg");
	CImg<int> img5 = canny5.canny_image(10.0, 0.25, 0.75);

	Canny canny6("./test_Data/lena.jpg", "jpg");
	CImg<int> img6 = canny6.canny_image(20.0, 0.25, 0.75);
	(img1, img2, img3, img4, img5, img6).display("Sigma Test");
}

void hysteresisMinValTest()
{
	Canny canny1("./test_Data/lena.jpg", "jpg");
	CImg<int> img1 = canny1.canny_image(2.0, 0.01, 0.75);

	Canny canny2("./test_Data/lena.jpg", "jpg");
	CImg<int> img2 = canny2.canny_image(2.0, 0.05, 0.75);

	Canny canny3("./test_Data/lena.jpg", "jpg");
	CImg<int> img3 = canny3.canny_image(2.0, 0.1, 0.75);

	Canny canny4("./test_Data/lena.jpg", "jpg");
	CImg<int> img4 = canny4.canny_image(2.0, 0.4, 0.75);

	Canny canny5("./test_Data/lena.jpg", "jpg");
	CImg<int> img5 = canny5.canny_image(2.0, 0.65, 0.75);

	Canny canny6("./test_Data/lena.jpg", "jpg");
	CImg<int> img6 = canny6.canny_image(2.0, 0.7, 0.75);
	(img1, img2, img3, img4, img5, img6).display("Hysteresis minVal Test");
}

void hysteresisMaxValTest()
{
	Canny canny1("./test_Data/lena.jpg", "jpg");
	CImg<int> img1 = canny1.canny_image(2.0, 0.25, 0.4);

	Canny canny2("./test_Data/lena.jpg", "jpg");
	CImg<int> img2 = canny2.canny_image(2.0, 0.25, 0.6);

	Canny canny3("./test_Data/lena.jpg", "jpg");
	CImg<int> img3 = canny3.canny_image(2.0, 0.25, 0.8);

	Canny canny4("./test_Data/lena.jpg", "jpg");
	CImg<int> img4 = canny4.canny_image(2.0, 0.25, 0.9);

	Canny canny5("./test_Data/lena.jpg", "jpg");
	CImg<int> img5 = canny5.canny_image(2.0, 0.25, 1.0);

	Canny canny6("./test_Data/lena.jpg", "jpg");
	CImg<int> img6 = canny6.canny_image(2.0, 0.25, 2.0);
	(img1, img2, img3, img4, img5, img6).display("Hysteresis maxVal Test");
}