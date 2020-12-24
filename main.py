import cv2
import time
import eventlet  # 导入eventlet这个模块
from OCR import ocr    # 引入ocr函数
import numpy as np
import sodoku_solver

suduko = []

for i in range(1,10):
    for j in range(1,10):
        img = cv2.imread('ImagesProcessing\\sudokus_slide\\sudoku1\\'+str(i)+'_'+str(j)+'.png',cv2.IMREAD_GRAYSCALE)
        # 设定分割阈值
        thresh_val = np.mean(img) - 25
        result = ocr.get_ocr_result(img,thresh_val)
        print(result,end=' ')
        suduko.append(result)
    print(' ')

# 数独求解
suduko = np.array(suduko)
suduko = np.reshape(suduko,(9,9))



eventlet.monkey_patch()  # 必须加这条代码
try:
    with eventlet.Timeout(2, True):  # 设置超时时间为2秒
        ans = sodoku_solver.solve(suduko)
        print(ans)
except eventlet.timeout.Timeout:
    print('数独无解')



