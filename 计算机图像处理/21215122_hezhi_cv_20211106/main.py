import numpy as np
import cv2
import matplotlib.pyplot as plt
from utils import *

# ex1:用 Canny 算子获取图像边缘点（结果请查看文件：./result/I_edge.jpg）
def ex1(imagePath):
    image = cv2.imread(imagePath, cv2.IMREAD_UNCHANGED)
    imageBlur = cv2.GaussianBlur(image, (3, 3), 1)
    imageGray = cv2.cvtColor(imageBlur, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(imageGray, 60, 400)
    
    imageOut = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    plt.imsave('./result/I_edge.jpg', imageOut)

    return edges

# ex2:计算名片边缘的各直线方程
def ex2(cannyImage):
    lines = cv2.HoughLinesP(cannyImage, 
                            1, 
                            np.pi / 180, 
                            threshold=80,
                            minLineLength=190,
                            maxLineGap=12)

    with open('./result/paramet_line_equation.txt', 'w') as f:
        imageLines = cv2.cvtColor(cannyImage, cv2.COLOR_GRAY2RGB)
        for i in range(lines.shape[0]):
            for x1, y1, x2, y2 in lines[i]:
                # a) 输出各个直线的参数方程(结果请查看文件：./result/aramet_line_equation.txt)
                equation = 'b = -k * {} + {}\n'.format(x1, y1)
                equation2 = 'b = -k * {} + {}\n'.format(x2, y2)
                f.write(equation)
                f.write(equation2)

                # b) 在上面的边缘图𝐈𝑒𝑑𝑔𝑒绘制直线, 用蓝色显示, 得到图像𝐈2（结果请查看文件：./result/I2.jpg）
                cv2.line(imageLines, (x1, y1), (x2, y2), (0, 0, 255), 3)

    imageLinesDots = imageLines.copy()
    for i in range(lines.shape[0]):
        for x1, y1, x2, y2 in lines[i]:
            # c) 在𝐈2图上显示 A4 纸的相关边缘点,用红色点显示, 得到图像𝐈3（结果请查看文件：./result/I3.jpg'）
            cv2.circle(imageLinesDots, (x1, y1), 2, (255, 0, 0), 4)
            cv2.circle(imageLinesDots, (x2, y2), 2, (244, 0, 0), 4)

    plt.imsave('./result/I2.jpg', imageLines)
    plt.imsave('./result/I3.jpg', imageLinesDots)
    
    return imageLinesDots, lines


# ex3: 输出名片纸的四个角点（结果请查看文件：./result/I4.jpg）
def ex3(sourceImage, lines):
    image4CrossPoints = sourceImage.copy()

    top, left, bottom, right = find4CrossPointWithLines(lines)
    leftTop = findCrossPointWithTwoLine(left, top)
    rightTop = findCrossPointWithTwoLine(right, top)
    leftBottom = findCrossPointWithTwoLine(left, bottom)
    rightBottom = findCrossPointWithTwoLine(right, bottom)

    cv2.circle(image4CrossPoints, (leftTop[0], leftTop[1]), 5, (0, 255, 0), 4)
    cv2.circle(image4CrossPoints, rightTop, 5, (0, 255, 0), 4)
    cv2.circle(image4CrossPoints, leftBottom, 5, (0, 255, 0), 4)
    cv2.circle(image4CrossPoints, rightBottom, 5, (0, 255, 0), 4)

    plt.imsave('./result/I4.jpg', image4CrossPoints)


if __name__ == "__main__":
    imagePath = './data/IMG_20200511_220813.jpg'
    edges = ex1(imagePath)
    imageEdgesDots, lines = ex2(edges)
    ex3(imageEdgesDots, lines)