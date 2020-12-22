import cv2
from OCR import ocr    # 引入ocr函数
import numpy as np
import sodoku_solver

suduko = []

for i in range(1,10):
    for j in range(1,10):
        img = cv2.imread('sudokus\\sudoku1\\'+str(i)+'_'+str(j)+'.png',cv2.IMREAD_GRAYSCALE)
        result = ocr.get_ocr_result(img)
        print(result,end=' ')
        suduko.append(result)
    print(' ')

suduko = np.array(suduko)
suduko = np.reshape(suduko,(9,9))
ans = sodoku_solver.solve(suduko)
print(ans)
