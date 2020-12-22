import cv2
from OCR import ocr    # 引入ocr函数
import numpy as np

for i in range(1,10):
    for j in range(1,10):
        img = cv2.imread('sudokus\\sudoku22\\'+str(i)+'_'+str(j)+'.png',cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img,(36,36))
        img = img[4:-4,4:-4]
        ret, img = cv2.threshold(img,150,255,cv2.THRESH_BINARY_INV)
        # cv2.imshow(str(i)+str(j), img) 
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        result = ocr.get_ocr_result(img)
        print(result,end=' ')
    print(' ')
