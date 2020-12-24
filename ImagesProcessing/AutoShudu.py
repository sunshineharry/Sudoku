# 自动取点代码
# coding=UTF-8<code>
from PIL import Image
from pylab import *
import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage
import os
import pickle
import sys


def get_process_control_number():
    """
    控制程序继续/终止
    :return: process_control_number: int,控制程序进程的标志
    """
    process_control_number = int(input('0---是\n1---否\n'))
    while process_control_number != 0 and process_control_number != 1:
        print('请输入有效数字！')
        process_control_number = int(input('0---是\n1---否\n'))
    return process_control_number


def get_points_autoly(img):
    """
    自动获取数独框的4个顶点
    :param img: 输入图像
    :return: points: 得到的数独框4个顶点
    """
    # 统计概率霍夫直线变换
    def line_detect_possible_demo(image):
        lines = cv2.HoughLinesP(image, 1, np.pi / 180, threshold=100, minLineLength=0, maxLineGap=100)
        # 创建保存霍夫变换直线检测结果的容器
        hough_img = np.ones(image.shape, dtype=np.uint8)
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(hough_img, (x1, y1), (x2, y2), (255, 255, 255), 10)
        # 进行概率霍夫直线检测
        ret, hough_img = cv2.threshold(hough_img, 127, 255, cv2.THRESH_BINARY)
        return hough_img
    # 边缘检测
    edges = cv2.Canny(img, 10, 60, apertureSize=3)
    # 腐蚀
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 30))  # 腐蚀矩阵
    corrosion_img = cv2.morphologyEx(edges, cv2.MORPH_DILATE, kernel)  # 腐蚀运算
    # 轮廓检测
    img_contour, contours, hierarchy = cv2.findContours(corrosion_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 得到最大矩形轮廓
    max_area = 0
    biggest_contour = None
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > max_area:
            max_area = area
            biggest_contour = cnt
    # 创建最大矩形轮廓的容器
    tmp_img = np.ones(img.shape, dtype=np.uint8)
    # 画出最大矩形轮廓
    tmp_img = cv2.drawContours(tmp_img, [biggest_contour], 0, (255, 255, 255), 30)
    # tmp_img = cv2.cvtColor(tmp_img, cv2.COLOR_RGB2GRAY)

    # 通过Sobel算子提取x方向的边缘，并做直线检测后做闭操作
    dx = cv2.Sobel(tmp_img, cv2.CV_16S, 1, 0)
    dx = cv2.convertScaleAbs(dx)
    cv2.normalize(dx, dx, 0, 255, cv2.NORM_MINMAX)
    ret, close = cv2.threshold(dx, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernelx = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 10))
    close = cv2.morphologyEx(close, cv2.MORPH_DILATE, kernelx, iterations=1)
    binary, contour, hierarchy = cv2.findContours(close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contour:
        x, y, w, h = cv2.boundingRect(cnt)
        if h / w > 5:
            cv2.drawContours(close, [cnt], 0, 255, -1)
        else:
            cv2.drawContours(close, [cnt], 0, 0, -1)

    close = cv2.morphologyEx(close, cv2.MORPH_CLOSE, None, iterations=1)
    closex = close.copy()
    line_closex = line_detect_possible_demo(closex)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 40))  # 腐蚀矩阵
    fushi_line_closex = cv2.morphologyEx(line_closex, cv2.MORPH_DILATE, kernel)  # 腐蚀运算

    # 通过Sobel算子提取y方向的边缘，并做直线检测后做闭操作
    dy = cv2.Sobel(tmp_img, cv2.CV_16S, 0, 1)
    dy = cv2.convertScaleAbs(dy)
    cv2.normalize(dy, dy, 0, 255, cv2.NORM_MINMAX)
    ret, close = cv2.threshold(dy, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernely = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 2))
    close = cv2.morphologyEx(close, cv2.MORPH_DILATE, kernely, iterations=1)
    binary, contour, hierarchy = cv2.findContours(close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contour:
        x, y, w, h = cv2.boundingRect(cnt)
        if w / h > 5:
            cv2.drawContours(close, [cnt], 0, 255, -1)
        else:
            cv2.drawContours(close, [cnt], 0, 0, -1)

    close = cv2.morphologyEx(close, cv2.MORPH_CLOSE, None, iterations=1)
    closey = close.copy()
    line_closey = line_detect_possible_demo(closey)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (100, 60))  # 腐蚀矩阵
    fushi_line_closey = cv2.morphologyEx(line_closey, cv2.MORPH_DILATE, kernel)  # 对腐蚀运算
    # 得到数独框的4个顶点
    res = cv2.bitwise_and(fushi_line_closex, fushi_line_closey)
    # 轮廓检测
    binary, contour, hierarchy = cv2.findContours(res, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    centroids = []
    for cnt in contour:
        if cv2.contourArea(cnt) > 20:
            mom = cv2.moments(cnt)
            (x, y) = int(mom['m10'] / mom['m00']), int(mom['m01'] / mom['m00'])
            centroids.append((x, y))
    centroids = np.array(centroids, dtype=np.float32)
    c = centroids.reshape((4, 2))
    c2 = c[np.argsort(c[:, 1])]
    # 得到角点
    b = np.vstack([c2[i * 10:(i + 1) * 10][np.argsort(c2[i * 10:(i + 1) * 10, 0])] for i in range(10)])
    bm = b.reshape((2, 2, 2))
    return bm


def perspective_trans(img, src_points):
    """
    透视变换
    :param img: 需要透视变换的图像
    :param src_points: 数独框顶点
    :return: perspective_trans_img: 透视变换后的图像
    """
    h, w = img.shape[:2]
    dst = np.array([[0, h], [0, 0], [w, 0], [w, h]], np.float32)
    src = np.array([src_points[0][0], src_points[0][1], src_points[1][0], src_points[1][1]], np.float32)
    # 得到透视变换的矩阵
    trans = cv2.getPerspectiveTransform(src, dst)
    perspective_trans_img = cv2.warpPerspective(img, trans, (w, h))
    return perspective_trans_img


def img_splite(img, img_number):
    """
    将数独图像分割为81个36*36的小图像
    :param img:
    :param img_number:
    :return: None
    """
    print('正在进行图像分割，请等待...')
    h, w = img.shape[:2]
    h_slice = int(h / 9)
    w_slice = int(w / 9)
    for i in range(0, 9):
        for j in range(0, 9):
            # 要被切割的开始的像素的高度值
            beH = h_slice * i
            # 要被切割的结束的像素的高度值
            hEnd = h_slice * (i + 1)
            # 要被切割的开始的像素的宽度值
            beW = w_slice * j
            # 要被切割的结束的像素的宽度值
            wLen = w_slice * (j + 1)
            dst_img = img[beH:hEnd, beW:wLen]
            # 存储图像
            dst_img = cv2.resize(dst_img, (36, 36))
            cv2.imwrite("../images/sudokus/sudoku{}/{}_{}.png".format(img_number, i + 1, j + 1), dst_img)
            # 提示运行结束
    print('已保存分割图像至文件夹sudoku{}，请至相应查验'.format(img_number))


def shadow_remove(img):
    """
    阴影去除
    :param img: 输入图像
    :return: shadow_remove_img: 输出的去除阴影后的图像
    """
    # 最大滤波
    def max_filtering(N, I_temp):
        wall = np.full((I_temp.shape[0] + (N // 2) * 2, I_temp.shape[1] + (N // 2) * 2), -1)
        wall[(N // 2):wall.shape[0] - (N // 2), (N // 2):wall.shape[1] - (N // 2)] = I_temp.copy()
        temp = np.full((I_temp.shape[0] + (N // 2) * 2, I_temp.shape[1] + (N // 2) * 2), -1)
        for y in range(0, wall.shape[0]):
            for x in range(0, wall.shape[1]):
                if wall[y, x] != -1:
                    window = wall[y - (N // 2):y + (N // 2) + 1, x - (N // 2):x + (N // 2) + 1]
                    num = np.amax(window)
                    temp[y, x] = num
        A = temp[(N // 2):wall.shape[0] - (N // 2), (N // 2):wall.shape[1] - (N // 2)].copy()
        return A

    # 最小滤波
    def min_filtering(N, A):
        wall_min = np.full((A.shape[0] + (N // 2) * 2, A.shape[1] + (N // 2) * 2), 300)
        wall_min[(N // 2):wall_min.shape[0] - (N // 2), (N // 2):wall_min.shape[1] - (N // 2)] = A.copy()
        temp_min = np.full((A.shape[0] + (N // 2) * 2, A.shape[1] + (N // 2) * 2), 300)
        for y in range(0, wall_min.shape[0]):
            for x in range(0, wall_min.shape[1]):
                if wall_min[y, x] != 300:
                    window_min = wall_min[y - (N // 2):y + (N // 2) + 1, x - (N // 2):x + (N // 2) + 1]
                    num_min = np.amin(window_min)
                    temp_min[y, x] = num_min
        B = temp_min[(N // 2):wall_min.shape[0] - (N // 2), (N // 2):wall_min.shape[1] - (N // 2)].copy()
        return B

    # 标准化
    def background_subtraction(I, B):
        O = I - B
        norm_img = cv2.normalize(O, None, 0, 255, norm_type=cv2.NORM_MINMAX)
        return norm_img

    # 先最小滤波，再最大滤波，进行阴影去除
    def min_max_filtering(M, N, I):
        if M == 0:
            # max_filtering
            A = max_filtering(N, I)
            # min_filtering
            B = min_filtering(N, A)
            # subtraction
            normalised_img = background_subtraction(I, B)
        elif M == 1:
            # min_filtering
            A = min_filtering(N, I)
            # max_filtering
            B = max_filtering(N, A)
            # subtraction
            normalised_img = background_subtraction(I, B)
        return normalised_img

    print('正在去除阴影，请等待...')
    # 阴影去除
    shadow_remove_img = min_max_filtering(M=0, N=20, I=img)
    # 改变类型
    shadow_remove_img = shadow_remove_img.astype(np.uint8)
    # 中值滤波器，消除椒盐噪声
    shadow_remove_img = cv2.medianBlur(shadow_remove_img, 5)
    print('去除阴影完成')
    return shadow_remove_img


def min_square_contour(img):
    """
    提取出数独框所在的最小外接四边形，对图像进行第一次正畸
    :param img: 获取图像
    :return: affine_trans_img: 仿射变换后的图像
    """
    # 提取图像边缘
    edges = cv2.Canny(img, 20, 100, apertureSize=3)
    # 腐蚀
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))  # 腐蚀矩阵
    corrosion_img = cv2.morphologyEx(edges, cv2.MORPH_DILATE, kernel)  # 腐蚀运算
    img_contour, contours, hierarchy = cv2.findContours(corrosion_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # 创建显示最小四边形的容器
    img_tmp = np.ones(img.shape, dtype=np.uint8)
    # 创建最小外接四边形顶点的容器
    boxes = []
    # 获取每个轮廓的最小外接四边形
    for c in contours:
        # 找面积最小的矩形
        rect = cv2.minAreaRect(c)
        # 得到最小矩形的坐标
        tmp_box = cv2.boxPoints(rect)
        # 标准化坐标到整数
        tmp_box = np.int0(tmp_box)
        boxes.append(tmp_box)
    # 根据轮廓面积从大到小排序
    boxes = sorted(boxes, key=cv2.contourArea, reverse=True)
    # 获取面积最大的最小外界四边形（认为此为数独框轮廓的最小外接四边形）
    cv2.drawContours(img_tmp, [boxes[0]], 0, (255, 255, 255), 10)
    print('最小外接四边形的顶点为：\n', boxes[0])
    print('如果输出的图像不正确，请首先确保顶点顺序依次应为左下角、左上角、右上角、右下角，否则需要对此函数获取仿射变换变换矩阵的部分进行修改！')
    # 下面进行仿射变换，进行对图像的第一次正畸
    # 获取原图的宽高
    h, w = img.shape[:2]
    dst = np.array([[0, h], [0, 0], [w, 0]], np.float32)
    src = np.array([boxes[0][0], boxes[0][1], boxes[0][2]], np.float32)
    # 获取变换矩阵
    trans = cv2.getAffineTransform(src, dst)
    # 仿射变换
    affine_trans_img = cv2.warpAffine(img, trans, (w, h))
    figure()
    title('数独框的第一次正畸')
    imshow(affine_trans_img, 'gray')
    print('关闭所有图像窗口后程序继续运行')
    show()
    return affine_trans_img



# ---------主程序----------
print("------Hello,python------")
print("手动提取数独数字")
# 解决PLT库中文显示问题
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# 选择读入的图像的编号
my_img_number = int(input('请输入1-24，读取相应编号的图像：\n'))
# 读入图像
original_img = cv2.imread('../images/sudokus/sudoku{}.jpg'.format(my_img_number))
# 变换图像为指定大小
original_img = cv2.resize(original_img, (2560, 1920))
# 显示图像
figure()
title('原图')
imshow(original_img)
print('关闭所有图像窗口后程序继续运行')
show()
# 图像预处理
# 中值滤波器，消除椒盐噪声
median_img = cv2.medianBlur(original_img, 5)
# 将图片数据类型转换为灰度图
gray_img = cv2.cvtColor(median_img, cv2.COLOR_RGB2GRAY)
figure()
title('去噪后的灰度图像')
imshow(gray_img, 'gray')
print('关闭所有图像窗口后程序继续运行')
show()
print('请选择是否需要去除阴影')
my_process_control_number = get_process_control_number()
if my_process_control_number == 0:
    gray_img = shadow_remove(gray_img)
    figure()
    title('去除阴影后的灰度图像')
    imshow(gray_img, 'gray')
    print('关闭所有图像窗口后程序继续运行')
    show()
# 获取数独框顶点
# 对图像进行第一次正畸
first_trans_img = min_square_contour(gray_img)
# 判断是否继续
print('图像是否正确？（是否为包含数独框的最小外接四边形？）')
my_process_control_number = get_process_control_number()
if my_process_control_number == 1:
    print('请修改程序或运行手动提取数独框程序')
    sys.exit()
# 获取数独框顶点
vertexes = get_points_autoly(first_trans_img)
print('数独框4个角点的坐标为：\n', vertexes)
print('如果输出的图像不正确，请首先确保角点顺序依次应为左下角、左上角、右上角、右下角，否则需要对此投射变换变换矩阵的部分进行修改！')
# 透视变换
trans_img = perspective_trans(first_trans_img, vertexes)
figure()
title('提取到的数独框')
imshow(trans_img, 'gray')
print('关闭所有图像窗口后程序继续运行')
show()
# 判断是否继续
print('图像是否正确？（是否为数独框？）')
my_process_control_number = get_process_control_number()
if my_process_control_number == 1:
    print('请修改程序或运行手动提取数独框程序')
    sys.exit()
# 图像分割
img_splite(trans_img, my_img_number)