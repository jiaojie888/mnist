from pathlib import Path
import sys
import numpy as np
import tensorflow as tf

# 添加项目根目录到模块搜索路径
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.model_loaders.logistic import LogisticRegression

# Load the dataset
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

# Preprocess the data
X_train = X_train.reshape(-1, 28 * 28).astype('float32') / 255.0
X_test = X_test.reshape(-1, 28 * 28).astype('float32') / 255.0

print("Training data shape:", X_train.shape)
print("Training labels shape:", y_train.shape)
print("Test data shape:", X_test.shape)
print("Test labels shape:", y_test.shape)

# new
model = LogisticRegression(
    learning_rate=0.1,
    epochs=10,
    batch_size=128
)

# train
model.fit(X_train, y_train)

# save
model.save()

# predict
predictions, confidences = model.predict(X_test)

accuracy = np.mean(predictions == y_test)
errors = np.sum(predictions != y_test)

print(f"测试准确率：{accuracy * 100:.2f}%")
print(f"错误数量：{errors}")
