# 手动取点代码
# coding=UTF-8<code>
from PIL import Image
from pylab import *
import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage
import os
import pickle


def get_points_manually(img):
    """
    手动获取数独框的4个顶点
    :param img: 输入图像
    :return: points: 得到的数独框4个顶点
    """
    # 改变这个参数调整对应点的对数
    point_pair = 4
    # 显示图像
    figure()
    title('选取数独框')
    imshow(img, 'gray')
    # 取点
    print('请点击{}个数独框顶点，依次应为左下角、左上角、右上角、右下角，请尽量准确选取'.format(point_pair))
    # 创建点的容器
    points = []
    for i in range(point_pair):
        x = ginput(1)
        print('第{}个:'.format(i + 1), x)
        points.append(x)
    return points


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


def perspective_trans(img, src_points):
    """
    透视变换
    :param img: 需要透视变换的图像
    :param src_points: 数独框顶点
    :return: perspective_trans_img: 透视变换后的图像
    """
    h, w = img.shape[:2]
    dst = np.array([[0, h], [0, 0], [w, 0], [w, h]], np.float32)
    # 对于实物，未检测出来的图像采取手动取点
    src = np.array([src_points[0], src_points[1], src_points[2], src_points[3]], np.float32)
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
vertexes = get_points_manually(gray_img)
# print(vertexes)
# 透视变换
trans_img = perspective_trans(gray_img, vertexes)
figure()
title('提取到的数独框')
imshow(trans_img, 'gray')
print('关闭所有图像窗口后程序继续运行')
show()
# 图像分割
img_splite(trans_img, my_img_number)
