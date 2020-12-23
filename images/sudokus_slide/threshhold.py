import cv2
import numpy as np

TEST = 1

for i in range(1,10):
    for j in range(1,10):
        imname = 'sudoku1\\'+str(i)+'_'+str(j)+'.png'
        img = cv2.imread(imname)
        img = cv2.resize(img,(36,36))
        img = img[4:-4,4:-4]
        # 参考minst数据集，进行二值处理并黑白反转
        ret, img = cv2.threshold(img,150,255,cv2.THRESH_BINARY_INV)
        if TEST:
            cv2.imshow("Image", img) 
            cv2.waitKey (0)
            cv2.destroyAllWindows()
        else:
            cv2.imwrite(imname,img)