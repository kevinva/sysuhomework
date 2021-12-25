/*
 * CannyEdgeDetection.h
 *
*/

#ifndef __CANNY_EDGE_DETECTION_H__
#define __CANNY_EDGE_DETECTION_H__

#include <windows.h>
#include <vector>

//----canny边缘检测-----------
class CCannyEdgeDetection
{
public:
    CCannyEdgeDetection();
    ~CCannyEdgeDetection();

public:
    int m_guid_index; //保存的图片格式，0-ImageFormatBMP,1-ImageFormatJPEG,2-ImageFormatPNG,3-ImageFormatGIF
    BITMAP m_bitmap;
    HBITMAP m_hBitmap;

public:
    //--------图片操作---------------
    int OpenImage(wchar_t* filename, BITMAP &bitmap, HBITMAP &hBitmap); //打开图片
    int SaveImage(wchar_t* filename, HBITMAP &hBitmap); //保存图片,针对位图句柄
    int SaveImage(wchar_t* filename, BITMAP &bitmap); //保存图片，针对内存中位图结果
    int CreateEmptyImage(BITMAP &bitmap, int width, int height, int bmBitsPixel); //在内存中创建一幅空白位图
    int ReleaseHandle(); //主动释放资源
    int ReleaseBitmap(BITMAP &bitmap); //主动释放资源

    int Canny(BITMAP &bitmap_src, BITMAP &bitmap_dst, double low_thresh, double high_thresh); //canny边缘检测
    int Sobel(BITMAP &bitmap_src, BITMAP &bitmap_dst, double low_thresh, double high_thresh); //Sobel图像一阶差分梯度
};

#endif //__CANNY_EDGE_DETECTION_H__
————————————————
版权声明：本文为CSDN博主「jfu22」的原创文章，遵循 CC 4.0 BY-SA 版权协议，转载请附上原文出处链接及本声明。
原文链接：https://blog.csdn.net/jfu22/article/details/50985651
