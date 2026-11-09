
from pathlib import Path
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "models" / "03_mlp" / "mlp.npz"


class MLPClassifier:
    def __init__(self, learning_rate=0.01, epochs=5):
        self.learning_rate = learning_rate
        self.epochs = epochs

        self.W1 = None
        self.b1 = None
        self.W2 = None
        self.b2 = None

    # 初始化参数
    def initialize(self):
        self.W1 = np.random.randn(784, 128) * 0.01
        self.b1 = np.zeros(128)

        self.W2 = np.random.randn(128, 10) * 0.01
        self.b2 = np.zeros(10)

    # 前向传播
    def forward(self, X):
        z1 = X @ self.W1 + self.b1
        h = np.maximum(0, z1)

        z2 = h @ self.W2 + self.b2

        exp_z2 = np.exp(z2 - np.max(z2, axis=1, keepdims=True))
        probabilities = exp_z2 / np.sum(
            exp_z2, axis=1, keepdims=True
        )

        return z1, h, probabilities

    # 训练单个样本
    def train_one(self, x, y):
        z1 = x @ self.W1 + self.b1
        h = np.maximum(0, z1)

        z2 = h @ self.W2 + self.b2

        exp_z2 = np.exp(z2 - np.max(z2))
        probabilities = exp_z2 / np.sum(exp_z2)

        loss = -np.log(probabilities[y] + 1e-12)

        # 反向传播
        Y = np.zeros(10)
        Y[y] = 1

        dz2 = probabilities - Y

        dW2 = np.outer(h, dz2)
        db2 = dz2

        dh = dz2 @ self.W2.T
        dz1 = dh * (z1 > 0)

        dW1 = np.outer(x, dz1)
        db1 = dz1

        # 更新参数
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1

        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2

        return loss

    # 训练模型
    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64).reshape(-1, 784)
        y = np.asarray(y, dtype=np.int64)

        self.initialize()

        for epoch in range(self.epochs):
            total_loss = 0

            for i in range(X.shape[0]):
                loss = self.train_one(X[i], y[i])
                total_loss += loss

            avg_loss = total_loss / X.shape[0]

            print(
                f"Epoch {epoch + 1}/{self.epochs}, "
                f"Loss: {avg_loss}"
            )

        return self

    # 保存模型
    def save(self):
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

        np.savez(
            MODEL_PATH,
            W1=self.W1,
            b1=self.b1,
            W2=self.W2,
            b2=self.b2
        )

        print("MLP 模型保存成功！")

    # 加载模型
    def load(self):
        data = np.load(MODEL_PATH)

        self.W1 = data["W1"]
        self.b1 = data["b1"]
        self.W2 = data["W2"]
        self.b2 = data["b2"]

        print("MLP 模型加载成功！")
        return self

    # 统一预测接口
    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)

        if X.ndim == 1:
            X = X.reshape(1, 784)
        elif X.ndim == 2 and X.shape == (28, 28):
            X = X.reshape(1, 784)
        elif X.ndim == 3:
            X = X.reshape(-1, 784)

        if X.ndim != 2 or X.shape[1] != 784:
            raise ValueError("输入图片维度不正确")

        _, _, probabilities = self.forward(X)

        predictions = np.argmax(probabilities, axis=1)
        confidences = np.max(probabilities, axis=1)

        if predictions.size == 1:
            return int(predictions[0]), float(confidences[0])

        return predictions, confidences
