
from pathlib import Path
import numpy as np


# 项目根目录
ROOT = Path(__file__).resolve().parents[2]

# 模型保存路径
MODEL_PATH = ROOT / "models" / "01_knn" / "knn.npz"


class KNNClassifier:
    def __init__(self, k=5):
        self.k = k
        self.X_train = None
        self.y_train = None

    # 接收训练数据
    def fit(self, X, y):
        self.X_train = np.asarray(X, dtype=np.float32)
        self.y_train = np.asarray(y, dtype=np.int64)

    # 保存模型
    def save(self):
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

        np.savez(
            MODEL_PATH,
            X_train=self.X_train,
            y_train=self.y_train
        )

        print("KNN 模型保存成功！")

    # 加载模型
    def load(self):
        data = np.load(MODEL_PATH)

        self.X_train = data["X_train"]
        self.y_train = data["y_train"]

        print("KNN 模型加载成功！")
        return self

    # 统一预测接口
    def predict(self, X, batch_size=100):
        X = np.asarray(X, dtype=np.float32)

        # 兼容单张图片 (784,)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        # 兼容单张图片 (28, 28)
        if X.ndim == 2 and X.shape == (28, 28):
            X = X.reshape(1, 784)

        # 兼容批量图片 (N, 28, 28)
        elif X.ndim == 3:
            X = X.reshape(X.shape[0], 784)

        if X.ndim != 2 or X.shape[1] != 784:
            raise ValueError("输入图片必须是 784 维，或 28×28 图片")

        train_norms = np.sum(self.X_train ** 2, axis=1)

        predictions = []
        confidences = []

        for i in range(0, X.shape[0], batch_size):
            batch = X[i:i + batch_size]

            batch_norms = np.sum(batch ** 2, axis=1)
            dot_product = batch @ self.X_train.T

            distances_squared = (
                batch_norms[:, np.newaxis]
                + train_norms[np.newaxis, :]
                - 2 * dot_product
            )

            distances_squared = np.maximum(distances_squared, 0)

            nearest_indices = np.argpartition(
                distances_squared,
                self.k - 1,
                axis=1
            )[:, :self.k]

            nearest_labels = self.y_train[nearest_indices]

            # 统计每张图片的投票
            for labels in nearest_labels:
                counts = np.bincount(labels, minlength=10)

                prediction = int(np.argmax(counts))
                confidence = float(counts[prediction] / self.k)

                predictions.append(prediction)
                confidences.append(confidence)

        # 单张图片返回一个预测结果和投票比例
        if len(predictions) == 1:
            return predictions[0], confidences[0]

        # 批量图片返回预测数组和投票比例数组
        return np.array(predictions), np.array(confidences)