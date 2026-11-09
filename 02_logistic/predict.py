import numpy as np
import tensorflow as tf
from PIL import Image


model = np.load("model_parameters.npz")

W1 = model['W1']
b1 = model['b1']
W2 = model['W2']
b2 = model['b2']

def predict(x):

    # 加载图像并进行预处理
    x = x.astype(np.float32).reshape(784) / 255.0

    # 前向传播
    z1 = x @ W1 + b1
    h = np.maximum(0, z1)

    z2 = h @ W2 + b2

    exp_z2 = np.exp(z2 - np.max(z2))
    probabilities = exp_z2 / np.sum(exp_z2)
    prediction = np.argmax(probabilities)

    return prediction, probabilities

# 测试模型
(_, _), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
prediction, probabilities = predict(X_test[0])
print("真实数字:", y_test[0])
print("预测数字:", prediction)
print("预测概率:", probabilities)