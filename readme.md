
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