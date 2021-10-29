#include "canny.h"

int main()
{
	Canny c;

	CImg<unsigned char> lena("../test_Data/lena.jpg");
	CImg<unsigned char> lenaCanny = c.build(lena, lena.width(), lena.height());
	lenaCanny.save("./result_Data/lena_canny.jpg");

	CImg<unsigned char> bigben("../test_Data/bigben.jpg");
	CImg<unsigned char> bigbenCanny = c.build(bigben, bigben.width(), bigben.height());
	bigbenCanny.save("./result_Data/bigben_canny.jpg");

	CImg<unsigned char> stpietro("../test_Data/stpietro.jpg");
	CImg<unsigned char> stpietroCanny = c.build(stpietro, stpietro.width(), stpietro.height());
	stpietroCanny.save("./result_Data/stpietro_canny.jpg");

	CImg<unsigned char> image1("../test_Data/3.jpg");
	CImg<unsigned char> image1Canny = c.build(image1, image1.width(), image1.height());
	image1Canny.save("./result_Data/3_canny.jpg");

	CImg<unsigned char> image2("../test_Data/4.jpg");
	CImg<unsigned char> image2Canny = c.build(image2, image2.width(), image2.height());
	image2Canny.save("./result_Data/4_canny.jpg");

	CImg<unsigned char> image3("../test_Data/20160326110137505.JPG");
	CImg<unsigned char> image3Canny = c.build(image3, image3.width(), image3.height());
	image3Canny.save("./result_Data/20160326110137505_canny.jpg");

	return 0;
}