import cv2
import numpy as np
# 根据线段端点计算对应直线交点，原理参考直线点斜式方程联立所解
def CrossPoint(line1, line2):
    x0, y0, x1, y1 = line1[0]
    x2, y2, x3, y3 = line2[0]

    dx1 = x1 - x0
    dy1 = y1 - y0

    dx2 = x3 - x2
    dy2 = y3 - y2

    D1 = x1 * y0 - x0 * y1
    D2 = x3 * y2 - x2 * y3

    y = float(dy1 * D2 - D1 * dy2) / (dy1 * dx2 - dx1 * dy2)
    x = float(y * dx1 - D1) / dy1

    return (int(x), int(y))


def SortPoint(points):
    sp = sorted(points, key=lambda x: (int(x[1]), int(x[0])))
    if sp[0][0] > sp[1][0]:
        sp[0], sp[1] = sp[1], sp[0]

    if sp[2][0] > sp[3][0]:
        sp[2], sp[3] = sp[3], sp[2]

    return sp


if __name__ == '__main__':

    src = cv2.imread("input.jpg")
    rgbsrc = src.copy()
    # .转成灰度图像
    graysrc = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)  # 将二值图转换为RGB图颜色空间，这里重新创建一张空Mat也行
    # 对灰度图做高斯滤波
    blurimg = cv2.GaussianBlur(graysrc, (3, 3), 0)
    # 获取边缘
    Cannyimg = cv2.Canny(blurimg, 35, 189)
    cv2.imshow("canny", Cannyimg);
    cv2.imwrite("outputCanny.jpg", Cannyimg)

    # 4. 霍夫变换检测 #提取边缘时，会造成有些点不连续，所以maxLineGap设大点
    lines = cv2.HoughLinesP(Cannyimg, 1, np.pi / 180, threshold=1, minLineLength=120, maxLineGap=10)
    # 蓝色颜色
    green = (255, 0, 0)
    # 5.显示检测到的直线，并画蓝色，指定粗细为5
    for line in lines:
        x1, y1, x2, y2 = line[0]
        dst = cv2.line(src, (x1, y1), (x2, y2), green, 5)

    cv2.imshow("Image", dst)
    cv2.waitKey(0)
    cv2.imwrite("output.jpg", dst)
    # 根据霍夫变换所获得的直线端点，用求直线交点的方式，求名片的四个角点


    # 根据上述霍夫变换获得的线段求直线交点，实验证明，霍夫变换获取并存储直线时是横纵方向依次完成的，即只需如下形式计算
    points = []
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            points.append(CrossPoint(lines[i], lines[j]))
    sp = SortPoint(points)
    # 红色颜色
    red = (0, 0, 255)
    count = 0
    pointFinall = []
    for point in points:
        x1 = point[0]
        y1 = point[1]
        count = count + 1  # 1,
        if count == 1 or count == 9 or count == 17 or count == 22:
            pointFinall.append(point)
            imageCircle = cv2.circle(src, (x1, y1), 20, (0, 0, 255), -1)
            cv2.putText(src,str(count),(x1, y1),cv2.FONT_HERSHEY_SIMPLEX,1.2, (255, 255, 255), 2)
            cv2.imwrite(str(count) + '.jpg', imageCircle)

    sp = SortPoint(pointFinall)

    width = int(np.sqrt(((sp[0][0] - sp[1][0]) ** 2) + (sp[0][1] - sp[1][1]) ** 2))
    height = int(np.sqrt(((sp[0][0] - sp[2][0]) ** 2) + (sp[0][1] - sp[2][1]) ** 2))

    dstrect = np.array([
        [0, 0],
        [width - 1, 0],
        [0, height - 1],
        [width - 1, height - 1]], dtype="float32")

    transform = cv2.getPerspectiveTransform(np.array(sp, dtype="float32"), dstrect)
    warpedimg = cv2.warpPerspective(rgbsrc, transform, (width, height))
    cv2.imshow("ImageCircle", warpedimg)
    cv2.waitKey(0)
    cv2.imwrite("outputWarpedimg.jpg", warpedimg)