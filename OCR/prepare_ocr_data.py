import os
import cv2
import numpy as np

imlist = [os.path.join('OCR\\OCR_data\\training',f) for f in os.listdir('OCR\\OCR_data\\training')]

for imname in imlist:
    img = cv2.imread(imname)
    img = cv2.resize(img,(36,36))
    img = img[4:-4,4:-4]
    cv2.imwrite(imname,img)