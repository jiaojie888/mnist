
from pathlib import Path
import numpy as np
import tensorflow as tf


class CNNTensorFlowModel:
    def __init__(self, model_path=None):
        root = Path(__file__).resolve().parents[2]

        if model_path is None:
            model_path = (
                root / "models" / "04_cnn" / "cnn_tensorflow.keras"
            )

        self.model_path = Path(model_path)
        self.model = None

    def load(self):
        """加载已经训练好的模型"""
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"模型文件不存在: {self.model_path}"
            )

        self.model = tf.keras.models.load_model(self.model_path)
        return self

    def predict(self, image):
        """
        image: 28×28 的 NumPy 数组
        返回: prediction, confidence
        """
        if self.model is None:
            raise RuntimeError("请先调用 load() 加载模型")

        image = np.asarray(image, dtype=np.float32)

        # 统一归一化到 0~1
        if image.max() > 1:
            image = image / 255.0

        # (28, 28) -> (1, 28, 28, 1)
        image = image.reshape(1, 28, 28, 1)

        probabilities = self.model.predict(
            image,
            verbose=0
        )[0]

        prediction = int(np.argmax(probabilities))
        confidence = float(probabilities[prediction])

        return prediction, confidence
