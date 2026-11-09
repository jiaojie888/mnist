import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# 0 设置支持中文显示
# 设置支持中文的字体（以 macOS 常用的苹方或黑体为例）
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang SC', 'Heiti TC']
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

# 1. 加载 MNIST 数据集
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

image = X_test[0]  # 选择测试集中的第一张图像

shifted_image = np.roll(image, shift=5, axis=1)  # 将图像向右平移5个像素

# 2. 显示原始图像和移动后的图像
plt.figure(figsize=(8, 4))

plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title("原始图像")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(shifted_image, cmap='gray')
plt.title("向右平移5个像素后的图像")
plt.axis('off')
plt.tight_layout()
plt.show()


distance = np.linalg.norm(image.flatten() - shifted_image.flatten())  # 计算原始图像和移动后的图像之间的欧氏距离
print(f"原始图像和移动后的图像之间的欧氏距离: {distance:.2f}")