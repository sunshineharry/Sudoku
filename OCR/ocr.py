from keras.models import load_model
import numpy as np

model = load_model('OCR\\number_ocr.h5')

def get_ocr_result(img):
    """Input the image to be predicted and return the prediction result

    # Arguments
        img: Ndarray, image to predict, which shape should be grayscale.

    # Returns
        num: the prediction of image.
    """

    # 数据维度尺寸变换
    img = img[4:-4,4:-4]
    img = img/255
    img = np.resize(img,(1,28,28,1))

    return model.predict_classes(img)[0]

# only for testing
if __name__ == "__main__":
    import cv2
    img = cv2.imread('OCR\\OCR_data\\testing\\0\\0_5.jpg' ,cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img,(36,36))
    result = get_ocr_result(img)
    print(result)