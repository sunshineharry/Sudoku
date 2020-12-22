import cv2
from OCR import ocr    # 引入ocr函数
import numpy as np

for i in range(1,10):
    for j in range(1,10):
        img = cv2.imread('sudokus\\sudoku6\\'+str(i)+'_'+str(j)+'.png',cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img,(36,36))
        img = img[4:-4,4:-4]
        tmp = 25*np.ones((28, 28),dtype=np.uint8)
        img = cv2.subtract(img,tmp)
        # cv2.imwrite('sudokus\\sudoku6\\'+str(i)+'_'+str(j)+'.png',img)
        # cv2.imshow(str(i)+str(j), img) 
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        result = ocr.get_ocr_result(img)
        print(result,end=' ')
    print(' ')
