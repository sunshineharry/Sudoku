from keras.models import load_model
import numpy as np
import cv2

model = load_model('OCR\\number_ocr.h5')

def get_ocr_result(img):

    # 数据维度尺寸变换
    img = cv2.resize(img,(36,36))
    img = img[4:-4,4:-4]
    ret, img = cv2.threshold(img,150,255,cv2.THRESH_BINARY_INV)
    img = img/255
    img = np.resize(img,(1,28,28,1))

    return model.predict_classes(img)[0]

if __name__ == "__main__":
    import cv2
    img = cv2.imread('OCR\\OCR_data\\testing\\0\\0_5.jpg' ,cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img,(36,36))
    result = get_ocr_result(img)
    print(result)