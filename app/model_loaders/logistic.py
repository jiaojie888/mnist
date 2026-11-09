
from pathlib import Path
import numpy as np


# 项目根目录
ROOT = Path(__file__).resolve().parents[2]

# 模型保存路径
MODEL_PATH = ROOT / "models" / "02_logistic" / "logistic.npz"


class LogisticRegression:
    def __init__(self, learning_rate=0.1, epochs=10, batch_size=128):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size

        self.W = None
        self.b = None

    # Softmax
    def softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    # 训练模型
    def fit(self, X, y):
        n_samples, n_features = X.shape
        n_classes = 10

        self.W = np.zeros((n_features, n_classes), dtype=np.float32)
        self.b = np.zeros(n_classes, dtype=np.float32)

        for epoch in range(self.epochs):
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            total_loss = 0

            for i in range(0, n_samples, self.batch_size):
                X_batch = X_shuffled[i:i + self.batch_size]
                y_batch = y_shuffled[i:i + self.batch_size]

                batch_size = X_batch.shape[0]

                # 前向传播
                z = X_batch @ self.W + self.b
                probabilities = self.softmax(z)

                # 计算交叉熵损失
                loss = -np.mean(
                    np.log(probabilities[np.arange(batch_size), y_batch] + 1e-12)
                )
                total_loss += loss * batch_size

                # 反向传播
                dz = probabilities.copy()
                dz[np.arange(batch_size), y_batch] -= 1
                dz /= batch_size

                dW = X_batch.T @ dz
                db = np.sum(dz, axis=0)

                # 更新参数
                self.W -= self.learning_rate * dW
                self.b -= self.learning_rate * db

            avg_loss = total_loss / n_samples
            print(f"Epoch {epoch + 1}/{self.epochs}, Loss: {avg_loss:.4f}")

        return self

    # 保存模型参数
    def save(self):
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

        np.savez(
            MODEL_PATH,
            W=self.W,
            b=self.b
        )

        print("逻辑回归模型保存成功！")

    # 加载模型参数
    def load(self):
        data = np.load(MODEL_PATH)

        self.W = data["W"]
        self.b = data["b"]

        print("逻辑回归模型加载成功！")
        return self

    # 统一预测接口
    def predict(self, X):
        X = np.asarray(X, dtype=np.float32)

        # 兼容单张 28×28 图片
        if X.ndim == 2 and X.shape == (28, 28):
            X = X.reshape(1, 784)

        # 兼容单张 784 维图片
        elif X.ndim == 1:
            X = X.reshape(1, 784)

        # 兼容批量 28×28 图片
        elif X.ndim == 3:
            X = X.reshape(X.shape[0], 784)

        if X.ndim != 2 or X.shape[1] != self.W.shape[0]:
            raise ValueError("输入图片维度不正确")

        # 前向传播
        z = X @ self.W + self.b
        probabilities = self.softmax(z)

        predictions = np.argmax(probabilities, axis=1)
        confidences = np.max(probabilities, axis=1)

        if predictions.size == 1:
            return int(predictions[0]), float(confidences[0])

        return predictions, confidences
