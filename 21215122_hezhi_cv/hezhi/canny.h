#include "CImg.h"

#ifndef CANNY_H
#define CANNY_H

using namespace cimg_library;

class Canny
{
public:
    CImg<unsigned char> build(CImg<unsigned char> greyImage, int width, int height);
	
private:
	CImg<unsigned char> image; //input
	int width;
	int height;
	int *edgesOutput;    // output
	int *magnitude;      // edge magnitude as detected by Gaussians
	float *xConv;        // temporary for convolution in x direction
	float *yConv;        // temporary for convolution in y direction
	float *dx;
	float *dy; 

	Canny *allocateBuffers(const CImg<unsigned char> & grey, int width, int height);
	void clearBuffers(Canny *canny);

    float gaussian(float x, float sigma);
	int computeGradients(Canny *canny, float kernelRaidus, int kernelWidth); 
	void performHysteresis(Canny *can, int low, int high);
	void follow(Canny *can, int x1, int y1, int i1, int threshold);
	void normalizeContrast(CImg<unsigned char> & data, int width, int height);
	float hypotenuse(float x, float y);
};

#endif