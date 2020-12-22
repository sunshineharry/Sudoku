# 版本信息

python==3.7

TensorFlow == 1.13.1

keras == 2.1.5

opencv == 3.4.2.16

# 接口示例
## OCR部分
接口使用示例代码，传入图像为`(28,28,1)`的灰度图像
```python
import cv2
from ORC import ocr    # 引入ocr函数

img = cv2.imread('OCR\\OCR_data\\testing\\0\\0_5.jpg',cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img,(28,28))
result = ocr.get_ocr_result(img)
print(result)
```

## 数独求解
接口使用示例代码，传入为(9,9)的矩阵，需要求解部分用`0`表示
```python
import sodoku_solver
import numpy as np

data =  "0 0 0 0 0 0 0 6 0 \
         0 0 0 0 0 4 7 0 5 \
         5 0 0 0 0 0 1 0 4 \
         1 0 0 0 0 2 4 0 0 \
         0 0 8 0 7 0 0 0 0 \
         0 3 0 6 0 0 0 0 0 \
         2 0 0 0 0 9 0 0 1 \
         0 0 6 0 8 0 0 0 0 \
         0 7 0 3 0 0 0 0 0 "
data = np.array(data.split(), dtype = int).reshape((9, 9))
print(data)
data = sodoku_solver.solve(data)
print('--------------------')
print(data)
```