import os
import cv2
from keras.preprocessing.image import ImageDataGenerator, array_to_img ,img_to_array, load_img
import numpy as np

TEST = 0

# 数据增强方式
datagen = ImageDataGenerator(
    rotation_range = 5,         # 随机旋转度数
    width_shift_range = 0.1,    # 随机水平平移
    height_shift_range = 0.1,   # 随机竖直平移
    shear_range = 5,            # 随机错切变换
    zoom_range = 0.1,           # 随机放大
    fill_mode = 'nearest',      # 填充方式
) 

# 训练集数据增强
for j in range(0,10):
    imlist = [os.path.join('OCR\\OCR_data\\training\\'+str(j),f) for f in os.listdir('OCR\\OCR_data\\training\\'+str(j))]

    for imname in imlist:
        img = cv2.imread(imname,cv2.IMREAD_GRAYSCALE)
        # 去除黑边
        img = cv2.resize(img,(36,36))
        img = img[4:-4,4:-4]
        # 参考minst数据集，进行二值处理并黑白反转
        ret, img = cv2.threshold(img,100,255,cv2.THRESH_BINARY_INV)


        if TEST:
            cv2.imshow("Image", img) 
            cv2.waitKey (0)
            cv2.destroyAllWindows()
        else:
            cv2.imwrite(imname,img)

            i = 0
            k = 0
            k += 1
            img = cv2.imread(imname)
            img = np.expand_dims(img,0)
            # 利用图像变换数据增强
            for batch in datagen.flow(img, batch_size=1, save_to_dir='OCR\\OCR_data\\training\\'+str(j), save_prefix='new_'+str(k), save_format='jpg'):
                i += 1
                if i==5:
                    break


for j in range(0,10):
    imlist = [os.path.join('OCR\\OCR_data\\testing\\'+str(j),f) for f in os.listdir('OCR\\OCR_data\\testing\\'+str(j))]

    for imname in imlist:
        img = cv2.imread(imname,cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img,(36,36))
        img = img[4:-4,4:-4]
        ret, img = cv2.threshold(img,100,255,cv2.THRESH_BINARY_INV)
        if TEST:
            cv2.imshow("Image", img) 
            cv2.waitKey (0)
            cv2.destroyAllWindows()
        else:
            cv2.imwrite(imname,img)