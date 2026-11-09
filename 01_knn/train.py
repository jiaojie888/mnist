import numpy as np
import tensorflow as tf
import sys
import time
from pathlib import Path

# 将项目根目录加入模块搜索路径
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# 导入公共 KNN 类
from app.model_loaders.knn import KNNClassifier


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


print("创建 KNN 分类器...")
knn = KNNClassifier(k=5)

# 3. 训练模型
print("训练模型...")
start_time = time.time()
knn.fit(X_train, y_train)
training_time = time.time() - start_time
print(f"训练时间: {training_time:.2f} 秒")


# 保存 KNN 模型
knn.save()

# predict
predictions = knn.predict(X_test)

accuracy = np.mean(predictions == y_test)
errors = np.sum(predictions != y_test)

print(f"测试准确率：{accuracy * 100:.2f}%")
print(f"错误数量：{errors}")
