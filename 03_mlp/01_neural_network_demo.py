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


print("W1:", W1.shape)
print("b1:", b1.shape)
print("W2:", W2.shape)
print("b2:", b2.shape)


# 取一张图片
x = X_train[0]

# 隐藏层
z1 = x @ W1 + b1
h = np.maximum(0, z1)

print("输入形状:", x.shape)
print("隐藏层形状:", h.shape)
print("原始计算结果:", z1[:10])
print("经过 ReLU:", h[:10])
print("隐藏层神经元总数:", len(h))
print("被 ReLU 变成 0 的数量:", np.sum(h == 0))

# 输出层
z2 = h @ W2 + b2

# softmax
exp_z2 = np.exp(z2 - np.max(z2))  # 减去最大值以防止溢出
probabilities = exp_z2 / np.sum(exp_z2)

print("输入形状:", x.shape)
print("隐藏层形状:", h.shape)
print("输出形状:", z2.shape)
# 查看结果
print("输出层分数:", z2)
print("预测概率:", probabilities)
print("概率总和:", np.sum(probabilities))
print("预测类别:", np.argmax(probabilities))
print("真实类别:", y_train[0])
print("Softmax 输出形状:", probabilities.shape)



y = y_train[0]

# 计算交叉熵损失
loss = -np.log(probabilities[y])

print("真实类别:", y)
print("交叉熵损失:", loss)
print("真实类别对应的概率:", probabilities[y])

# 构造 One-hot 真实标签
Y = np.zeros(10)
Y[y] = 1

# 计算输出层梯度
dz2 = probabilities - Y

print("预测概率:", probabilities)
print("真实标签:", Y)
print("输出层梯度:", dz2)

# 计算输出层参数的梯度
dW2 = np.outer(h, dz2)
db2 = dz2

print("dW2 形状:", dW2.shape)
print("db2 形状:", db2.shape)

# 计算隐藏层梯度
dh = dz2 @ W2.T

# 计算 ReLU 的梯度
dz1 = dh * (z1 > 0)
print("隐藏层梯度:", dz1)

# 计算隐藏层参数的梯度
dW1 = np.outer(x, dz1)
db1 = dz1


# 更新参数
learning_rate = 0.01

W1 -= learning_rate * dW1
b1 -= learning_rate * db1

W2 -= learning_rate * dW2
b2 -= learning_rate * db2
