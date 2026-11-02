import numpy as np
import tensorflow as tf
import time
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# 1. 加载 MNIST 数据集
print("加载 MNIST 数据集...")
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
print("MNIST 数据集加载完成.")

# 2. 数据预处理
print("数据预处理...")
X_train = X_train.reshape(X_train.shape[0], -1) / 255.0  # 展平并归一化
X_test = X_test.reshape(X_test.shape[0], -1) / 255.0  # 展平并归一化
print("数据预处理完成.")

print("训练集样本:", X_train.shape)
print("测试集样本:", X_test.shape)

class KNNClassifier:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict_one(self, x):
        # 计算欧氏距离 
        distances = np.linalg.norm(self.X_train - x, axis=1)
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = self.y_train[k_indices]
        most_common = np.bincount(k_nearest_labels).argmax()
        return most_common

    def predict(self, X):
        return np.array([self.predict_one(x) for x in X])


print("创建 KNN 分类器...")
knn = KNNClassifier(k=5)

# 3. 训练模型
print("训练模型...")
start_time = time.time()
knn.fit(X_train, y_train)
training_time = time.time() - start_time
print(f"训练时间: {training_time:.2f} 秒")

# 4. 在测试集上进行预测
print("在测试集上进行预测...")
start_time = time.time()
y_pred = knn.predict(X_test)
prediction_time = time.time() - start_time
print(f"预测时间: {prediction_time:.2f} 秒")

# 5. 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print("========= 实验结果 =========")
print(f"使用的 k 值: {knn.k}")
print(f"训练时间: {training_time:.2f} 秒")
print(f"预测时间: {prediction_time:.2f} 秒")
print(f"准确率: {accuracy:.2f}")
print("错误分类的样本数量:", np.sum(y_test != y_pred))
print("============================")

print("显示前10个错误分类的样本...")
wrong_indices = np.where(y_test != y_pred)[0]
plt.figure(figsize=(12, 5))
for i, idx in enumerate(wrong_indices[:10]):
    plt.subplot(2, 5, i + 1)
    plt.imshow(X_test[idx].reshape(28, 28), cmap='gray')
    plt.title(f"True: {y_test[idx]}, Pred: {y_pred[idx]}")
    plt.axis('off')
plt.tight_layout()
plt.show()
