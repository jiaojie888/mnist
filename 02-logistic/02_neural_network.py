import numpy as np
import tensorflow as tf

# 加载 MNIST
(X_train, y_train), (X_test, y_test) = (
    tf.keras.datasets.mnist.load_data()
)

# 归一化
X_train = X_train.astype(np.float32) / 255.0
X_test = X_test.astype(np.float32) / 255.0

# 展平图片
X_train = X_train.reshape(-1, 784)
X_test = X_test.reshape(-1, 784)

# 初始化参数
W1 = np.random.randn(784, 128) * 0.01
b1 = np.zeros(128)

W2 = np.random.randn(128, 10) * 0.01
b2 = np.zeros(10)

# 取一张图片
x = X_train[0]

# 隐藏层
z1 = x @ W1 + b1
h = np.maximum(0, z1)

# 输出层
z2 = h @ W2 + b2

print("输入形状:", x.shape)
print("隐藏层形状:", h.shape)
print("输出形状:", z2.shape)