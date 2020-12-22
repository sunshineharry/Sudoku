import os
import cv2
import numpy as np

for j in range(0,10):
    imlist = [os.path.join('OCR\\OCR_data\\training\\'+str(j),f) for f in os.listdir('OCR\\OCR_data\\training\\'+str(j))]

    for imname in imlist:
        img = cv2.imread(imname,cv2.IMREAD_GRAYSCALE)
        # img = cv2.resize(img,(36,36))
        # img = img[4:-4,4:-4]
        ret, img = cv2.threshold(img,127,255,cv2.THRESH_BINARY_INV)
        # cv2.imshow("Image", img) 
        # cv2.waitKey (0)
        # cv2.destroyAllWindows()
        cv2.imwrite(imname,img)