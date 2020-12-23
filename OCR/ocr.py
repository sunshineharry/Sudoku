from keras.models import load_model
import numpy as np
import cv2

model = load_model('OCR\\number_ocr.h5')

TEST = False

def get_ocr_result(img,thresh_val):

    # 数据维度尺寸变换
    img = cv2.resize(img,(36,36))
    img = img[4:-4,4:-4]
    ret, img = cv2.threshold(img,thresh_val,255,cv2.THRESH_BINARY_INV)
    if TEST:
        cv2.imshow("Image", img) 
        cv2.waitKey (0)
        cv2.destroyAllWindows()
    img = img/255
    img = np.resize(img,(1,28,28,1))

    return model.predict_classes(img)[0]