import cv2
from OCR import ocr    # 引入ocr函数

for i in range(1,10):
    for j in range(1,10):
        img = cv2.imread('sudokus\\sudoku16\\'+str(i)+'_'+str(j)+'.png',cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img,(28,28))
        result = ocr.get_ocr_result(img)
        print(result,end=' ')
    print(' ')
