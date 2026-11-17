
from pathlib import Path
import sys
import numpy as np
import tensorflow as tf

# 添加项目根目录
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.model_loaders.cnn import CNNModel


# ==========================================
# 1. 加载 MNIST 数据
# ==========================================

(X_train, y_train), (X_test, y_test) = (
    tf.keras.datasets.mnist.load_data()
)

X_train = X_train.astype(np.float32) / 255.0
X_test = X_test.astype(np.float32) / 255.0

# 增加通道维度
X_train = X_train[..., np.newaxis]
X_test = X_test[..., np.newaxis]

print("训练集:", X_train.shape)
print("测试集:", X_test.shape)


# ==========================================
# 2. 创建模型
# ==========================================

model = CNNModel(
    epochs=5,
    batch_size=32,
    learning_rate=0.01
)


# ==========================================
# 3. 训练模型
# ==========================================

# 先使用5000张图片
model.fit(
    X_train[:5000],
    y_train[:5000]
)


# ==========================================
# 4. 测试模型
# ==========================================

test_size = 1000

correct = 0

for i in range(test_size):

    prediction, confidence = model.predict(X_test[i])

    if prediction == y_test[i]:
        correct += 1

accuracy = correct / test_size

print(f"测试数量: {test_size}")
print(f"测试准确率: {accuracy:.2%}")
print(f"错误数量: {test_size - correct}")


# ==========================================
# 5. 保存模型
# ==========================================

model.save()

print("CNN 训练完成！")
