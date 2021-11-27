import cv2
import numpy as np


# 计算交点函数
def cross_point(rho_1, theta_1, rho_2, theta_2):
    a_h = np.cos(theta_1)
    b_h = np.sin(theta_1)
    x_h = a_h * rho_1
    y_h = b_h * rho_1
    x_1 = int(x_h + 1500 * (-b_h))
    y_1 = int(y_h + 1500 * a_h)
    x_2 = int(x_h - 1500 * (-b_h))
    y_2 = int(y_h - 1500 * a_h)
    # 取直线坐标两点的x和y值

    a_v = np.cos(theta_2)
    b_v = np.sin(theta_2)
    x_v = a_v * rho_2
    y_v = b_v * rho_2
    x_3 = int(x_v + 1500 * (-b_v))
    y_3 = int(y_v + 1500 * a_v)
    x_4 = int(x_v - 1500 * (-b_v))
    y_4 = int(y_v - 1500 * a_v)

    # L2直线斜率不存在操作
    if (x_4 - x_3) == 0:
        k2 = None
        b2 = 0
        x = x_3
        # 计算k1,由于点均为整数，需要进行浮点数转化
        k1 = (y_2 - y_1) * 1.0 / (x_2 - x_1)
        # 整型转浮点型是关键
        b1 = y_1 * 1.0 - x_1 * k1 * 1.0
        y = k1 * x * 1.0 + b1 * 1.0
    elif (x_2 - x_1) == 0:
        k1 = None
        b1 = 0
        x = x_1
        k2 = (y_4 - y_3) * 1.0 / (x_4 - x_3)
        b2 = y_3 * 1.0 - x_3 * k2 * 1.0
        y = k2 * x * 1.0 + b2 * 1.0
    else:
        # 计算k1,由于点均为整数，需要进行浮点数转化
        k1 = (y_2 - y_1) * 1.0 / (x_2 - x_1)
        # 斜率存在操作
        k2 = (y_4 - y_3) * 1.0 / (x_4 - x_3)
        # 整型转浮点型是关键
        b1 = y_1 * 1.0 - x_1 * k1 * 1.0
        b2 = y_3 * 1.0 - x_3 * k2 * 1.0
        x = (b2 - b1) * 1.0 / (k1 - k2)
        y = k1 * x * 1.0 + b1 * 1.0
    return x, y


# 读取原始图片
src_img = cv2.imread('./data/pic_02.jpg', -1)
gray_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)
edges_img = cv2.Canny(gray_img, 100, 280, 10)
cv2.imwrite('./picture/edges_img.jpg', edges_img)

# lines = cv2.HoughLines(edges_img, 1, np.pi / 180, 155) 调整第四个参数，使得水平与垂直两个方向出现的直线尽量少些
lines = cv2.HoughLines(edges_img, 1, np.pi / 180, 202)
lines_2d = lines[:, 0, :]
lines_horizontal = []
lines_vertical = []
for rho, theta in lines_2d[:]:
    if (theta < (np.pi / 4.)) or (theta > (3. * np.pi / 4.0)):
        lines_vertical.append([rho, theta])
    if (theta > (np.pi / 4.)) and (theta < (3. * np.pi / 4.0)):
        lines_horizontal.append([rho, theta])
# 获取水平与垂直的两条直线极坐标信息
lines_horizontal_2 = sorted(lines_horizontal)[0::len(lines_horizontal) - 1]
lines_vertical_2 = sorted(lines_vertical)[0::len(lines_vertical) - 1]
# 画直线
for rho, theta in lines_2d[:]:
    if [rho, theta] in lines_vertical_2 or [rho, theta] in lines_horizontal_2:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1500 * (-b))
        y1 = int(y0 + 1500 * a)
        x2 = int(x0 - 1500 * (-b))
        y2 = int(y0 - 1500 * a)
        cv2.line(src_img, (x1, y1), (x2, y2), (255, 0, 0), 1)
cv2.imwrite('./picture/lines_img.jpg', src_img)

# 画交点
srcPt = []
for rho_h, theta_h in lines_horizontal_2[:]:
    for rho_v, theta_v in lines_vertical_2[:]:
        crossP = cross_point(rho_h, theta_h, rho_v, theta_v)
        srcPt.append([crossP[0], crossP[1]])
        cv2.circle(src_img, (int(crossP[0]), int(crossP[1])), 10, (0, 0, 255), thickness=2)
cv2.imwrite('./picture/circles_img.jpg', src_img)
# 打印交点
print(srcPt)

# 校正图片
src_pt = np.float32(srcPt)
dst_pt = np.float32(
    [[srcPt[2][0], srcPt[1][1]], [srcPt[1][0], srcPt[1][1]], [srcPt[2][0], srcPt[2][1]], [srcPt[1][0], srcPt[2][1]]])
mat = cv2.getPerspectiveTransform(src_pt, dst_pt)
# 进行透视变换
warp_img = cv2.warpPerspective(src_img, mat, tuple(reversed(gray_img.shape)))
cv2.imwrite('./picture/warp_img.jpg', warp_img)
# 截取图片
cv2.imwrite("./picture/cut_img.jpg", warp_img[int(srcPt[1][0]):int(srcPt[2][1]), int(srcPt[1][1]):int(srcPt[2][0])])
cv2.waitKey(0)
cv2.destroyAllWindows()
