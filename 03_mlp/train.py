
from pathlib import Path
import sys
import numpy as np
import tensorflow as tf

# 添加项目根目录
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.model_loaders.mlp import MLPClassifier


# 加载 MNIST
(X_train, y_train), (X_test, y_test) = (
    tf.keras.datasets.mnist.load_data()
)

# 数据预处理
X_train = X_train.astype(np.float32) / 255.0
X_test = X_test.astype(np.float32) / 255.0

X_train = X_train.reshape(-1, 784)
X_test = X_test.reshape(-1, 784)


# 创建并训练模型
model = MLPClassifier(
    learning_rate=0.01,
    epochs=5
)

model.fit(X_train, y_train)

# 保存模型
model.save()


# 测试模型
predictions, confidences = model.predict(X_test)

accuracy = np.mean(predictions == y_test)
errors = np.sum(predictions != y_test)

print(f"测试准确率：{accuracy * 100:.2f}%")
print(f"错误数量：{errors}")