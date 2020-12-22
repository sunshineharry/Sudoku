# OCR训练与预测

## 训练模型

- 使用数据增强，对图像进行随机旋转，随机平移，错切，放大等方式，增强数据集

- 采用`keras`建立模型，通过两层卷积核两层池化提取特征，通过两层全连接完成类别的判定，在其中，引入`dropout`机制，防止过拟合，神经网络结构如下，共3,274,634个参数

<img src="C:\Users\Harry\OneDrive - bjtu.edu.cn\学习资料\大三上学期\机器视觉\大作业\OCR\readme_img\Snipaste_2020-12-20_16-40-02.jpg" alt="Snipaste_2020-12-20_16-40-02" style="zoom:50%;" />

- 训练结果：在训练集上准确度99.65%，测试集上准确度100%。

![Snipaste_2020-12-20_18-43-36](C:\Users\Harry\OneDrive - bjtu.edu.cn\学习资料\大三上学期\机器视觉\大作业\OCR\readme_img\Snipaste_2020-12-20_18-43-36.jpg)

