import numpy as np
import cv2
import matplotlib.pyplot as plt
from utils import *

DATA_DIR = './data'
OUT_DIR = './result'

# ex1:用 Canny 算子获取图像边缘点（结果请查看文件：./result/I_edge.jpg）
def ex1(imagePath):
    image = cv2.imread(imagePath, cv2.IMREAD_UNCHANGED)
    imageBlur = cv2.GaussianBlur(image, (3, 3), 1)
    imageGray = cv2.cvtColor(imageBlur, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(imageGray, 60, 400)
    
    imageOut = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    plt.imsave(OUT_DIR + '/' + 'I_edge.jpg', imageOut)

    return image, edges


# ex2:计算名片边缘的各直线方程，且输入如下结果：
def ex2(cannyImage):
    lines = cv2.HoughLinesP(cannyImage, 
                            1, 
                            np.pi / 180, 
                            threshold=80,
                            minLineLength=190,
                            maxLineGap=12)

    with open(OUT_DIR + '/' + 'paramet_line_equation.txt', 'w') as f:
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

    plt.imsave(OUT_DIR + '/' + 'I2.jpg', imageLines)
    plt.imsave(OUT_DIR + '/' + 'I3.jpg', imageLinesDots)
    
    return imageLinesDots, lines


# ex3: 输出名片纸的四个角点（结果请查看文件：./result/I4.jpg）
def ex3(sourceImage, lines):
    image4CrossPoints = sourceImage.copy()

    top, left, bottom, right = find4CrossPointWithLines(lines)
    leftTop = findCrossPointWithTwoLine(left, top)
    rightTop = findCrossPointWithTwoLine(right, top)
    leftBottom = findCrossPointWithTwoLine(left, bottom)
    rightBottom = findCrossPointWithTwoLine(right, bottom)

    cv2.circle(image4CrossPoints, leftTop, 5, (0, 255, 0), 4)
    cv2.circle(image4CrossPoints, rightTop, 5, (0, 255, 0), 4)
    cv2.circle(image4CrossPoints, leftBottom, 5, (0, 255, 0), 4)
    cv2.circle(image4CrossPoints, rightBottom, 5, (0, 255, 0), 4)

    plt.imsave(OUT_DIR + '/' + 'I4.jpg', image4CrossPoints)

    return leftTop, rightTop, leftBottom, rightBottom


# ex4: 已经矫正好的标准普通名片（输入效果请查看：./result/final.jpg）
def ex4(sourceImage, crossPoints):
    leftTop, rightTop, leftBottom, rightBottom = crossPoints
    points1 = [[leftTop[0], leftTop[1]], 
               [rightTop[0], rightTop[1]], 
               [leftBottom[0], leftBottom[1]], 
               [rightBottom[0], rightBottom[1]]]
    points1 = np.array(points1, dtype=np.float32)

    width = np.sqrt(np.power(rightTop[0] - leftTop[0], 2) + np.power(rightTop[1] - leftTop[1], 2))
    height = np.sqrt(np.power(leftBottom[0]- leftTop[0], 2) + np.power(leftBottom[1] - leftTop[1], 2))
    width = np.int16(width)
    height = np.int16(height)
    points2 = [[0, 0], [width, 0], [0, height], [width, height]]
    points2 = np.array(points2, dtype=np.float32)

    matrix = cv2.getPerspectiveTransform(points1, points2)
    imageOut = cv2.warpPerspective(sourceImage, matrix, (width, height))
    imageOut = cv2.cvtColor(imageOut, cv2.COLOR_BGR2RGB)
    plt.imsave(OUT_DIR + '/' + 'final.jpg', imageOut)




if __name__ == "__main__":
    imagePath = DATA_DIR + '/' + 'IMG_20200511_220813.jpg'
    # imagePath = DATA_DIR + '/' + 'IMG_20200511_220758.jpg'
    # imagePath = DATA_DIR + '/' + 'IMG_20160210_103112.jpg'
    sourceImage, edges = ex1(imagePath)
    imageEdgesDots, lines = ex2(edges)
    leftTop, rightTop, leftBottom, rightBottom = ex3(imageEdgesDots, lines)
    ex4(sourceImage, (leftTop, rightTop, leftBottom, rightBottom))