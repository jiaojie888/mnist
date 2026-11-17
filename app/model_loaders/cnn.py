
from pathlib import Path
import numpy as np


# 项目根目录
ROOT = Path(__file__).resolve().parents[2]


class CNNModel:

    def __init__(
        self,
        epochs=5,
        batch_size=32,
        learning_rate=0.01,
        model_path=None
    ):
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        if model_path is None:
            model_path = ROOT / "models" / "04_cnn" / "cnn.npz"

        self.model_path = Path(model_path)

        self.rng = np.random.default_rng(42)

        self._initialize_parameters()

    # ==========================================
    # 1. 初始化参数
    # ==========================================

    def _initialize_parameters(self):

        # Conv1: 1 -> 8
        self.kernels1 = (
            self.rng.standard_normal((8, 3, 3, 1))
            .astype(np.float32) * 0.1
        )
        self.biases1 = np.zeros(8, dtype=np.float32)

        # Conv2: 8 -> 16
        self.kernels2 = (
            self.rng.standard_normal((16, 3, 3, 8))
            .astype(np.float32) * 0.1
        )
        self.biases2 = np.zeros(16, dtype=np.float32)

        # Dense1: 400 -> 128
        self.W1 = (
            self.rng.standard_normal((400, 128))
            .astype(np.float32)
            * np.sqrt(2.0 / 400)
        )
        self.b1 = np.zeros(128, dtype=np.float32)

        # Dense2: 128 -> 10
        self.W2 = (
            self.rng.standard_normal((128, 10))
            .astype(np.float32)
            * np.sqrt(2.0 / 128)
        )
        self.b2 = np.zeros(10, dtype=np.float32)

    # ==========================================
    # 2. 卷积前向传播
    # ==========================================

    def conv2d(self, image, kernels, biases):

        image_h, image_w, _ = image.shape

        num_filters, kernel_h, kernel_w, _ = kernels.shape

        output_h = image_h - kernel_h + 1
        output_w = image_w - kernel_w + 1

        output = np.zeros(
            (output_h, output_w, num_filters),
            dtype=np.float32
        )

        for f in range(num_filters):
            for i in range(output_h):
                for j in range(output_w):

                    region = image[
                        i:i + kernel_h,
                        j:j + kernel_w,
                        :
                    ]

                    output[i, j, f] = (
                        np.sum(region * kernels[f]) + biases[f]
                    )

        return output

    # ==========================================
    # 3. 卷积反向传播
    # ==========================================

    def conv2d_backward(self, image, kernels, d_output):

        num_filters, kernel_h, kernel_w, _ = kernels.shape

        output_h, output_w, _ = d_output.shape

        d_image = np.zeros_like(image)
        d_kernels = np.zeros_like(kernels)
        d_biases = np.zeros(num_filters, dtype=np.float32)

        for f in range(num_filters):
            for i in range(output_h):
                for j in range(output_w):

                    region = image[
                        i:i + kernel_h,
                        j:j + kernel_w,
                        :
                    ]

                    grad = d_output[i, j, f]

                    d_kernels[f] += region * grad

                    d_image[
                        i:i + kernel_h,
                        j:j + kernel_w,
                        :
                    ] += kernels[f] * grad

                    d_biases[f] += grad

        return d_image, d_kernels, d_biases

    # ==========================================
    # 4. Max Pooling 前向传播
    # ==========================================

    def max_pool2d_forward(
        self,
        feature_map,
        pool_size=2,
        stride=2
    ):

        height, width, channels = feature_map.shape

        output_h = (height - pool_size) // stride + 1
        output_w = (width - pool_size) // stride + 1

        output = np.zeros(
            (output_h, output_w, channels),
            dtype=np.float32
        )

        max_indices = np.zeros(
            (output_h, output_w, channels),
            dtype=np.int32
        )

        for c in range(channels):
            for i in range(output_h):
                for j in range(output_w):

                    h_start = i * stride
                    w_start = j * stride

                    region = feature_map[
                        h_start:h_start + pool_size,
                        w_start:w_start + pool_size,
                        c
                    ]

                    max_index = np.argmax(region)

                    output[i, j, c] = region.flat[max_index]
                    max_indices[i, j, c] = max_index

        return output, max_indices

    # ==========================================
    # 5. Max Pooling 反向传播
    # ==========================================

    def max_pool2d_backward(
        self,
        d_output,
        max_indices,
        input_shape,
        pool_size=2,
        stride=2
    ):

        d_input = np.zeros(input_shape, dtype=np.float32)

        output_h, output_w, channels = d_output.shape

        for c in range(channels):
            for i in range(output_h):
                for j in range(output_w):

                    h_start = i * stride
                    w_start = j * stride

                    max_index = max_indices[i, j, c]

                    max_row, max_col = np.unravel_index(
                        max_index,
                        (pool_size, pool_size)
                    )

                    d_input[
                        h_start + max_row,
                        w_start + max_col,
                        c
                    ] += d_output[i, j, c]

        return d_input

    # ==========================================
    # 6. 前向传播
    # ==========================================

    def forward(self, image):

        cache = {}

        # Conv1 + ReLU + Pool1
        z1 = self.conv2d(
            image,
            self.kernels1,
            self.biases1
        )

        a1 = np.maximum(0, z1)

        pool1, pool1_indices = self.max_pool2d_forward(a1)

        # Conv2 + ReLU + Pool2
        z2 = self.conv2d(
            pool1,
            self.kernels2,
            self.biases2
        )

        a2 = np.maximum(0, z2)

        pool2, pool2_indices = self.max_pool2d_forward(a2)

        # Flatten
        flattened = pool2.reshape(-1)

        # Dense1 + ReLU
        z3 = flattened @ self.W1 + self.b1

        hidden = np.maximum(0, z3)

        # Dense2
        logits = hidden @ self.W2 + self.b2

        # Softmax
        exp_logits = np.exp(logits - np.max(logits))
        probabilities = exp_logits / np.sum(exp_logits)

        # 缓存反向传播需要的数据
        cache["image"] = image

        cache["z1"] = z1
        cache["pool1"] = pool1
        cache["pool1_indices"] = pool1_indices

        cache["z2"] = z2
        cache["pool2"] = pool2
        cache["pool2_indices"] = pool2_indices

        cache["flattened"] = flattened
        cache["z3"] = z3
        cache["hidden"] = hidden

        return probabilities, cache

    # ==========================================
    # 7. 反向传播
    # ==========================================

    def backward(self, probabilities, label, cache):

        # Softmax + Cross Entropy
        Y = np.zeros(10, dtype=np.float32)
        Y[label] = 1.0

        dz4 = probabilities - Y

        # Dense2
        dW2 = np.outer(cache["hidden"], dz4)
        db2 = dz4

        # Dense1
        dhidden = dz4 @ self.W2.T

        dz3 = dhidden * (cache["z3"] > 0)

        dW1 = np.outer(cache["flattened"], dz3)
        db1 = dz3

        dflattened = dz3 @ self.W1.T

        dpool2 = dflattened.reshape(cache["pool2"].shape)

        # Pool2 backward
        da2 = self.max_pool2d_backward(
            dpool2,
            cache["pool2_indices"],
            cache["z2"].shape
        )

        # ReLU2 backward
        dz2 = da2 * (cache["z2"] > 0)

        # Conv2 backward
        dpool1, d_kernels2, d_biases2 = self.conv2d_backward(
            cache["pool1"],
            self.kernels2,
            dz2
        )

        # Pool1 backward
        da1 = self.max_pool2d_backward(
            dpool1,
            cache["pool1_indices"],
            cache["z1"].shape
        )

        # ReLU1 backward
        dz1 = da1 * (cache["z1"] > 0)

        # Conv1 backward
        dimage, d_kernels1, d_biases1 = self.conv2d_backward(
            cache["image"],
            self.kernels1,
            dz1
        )

        return {
            "dW1": dW1,
            "db1": db1,
            "dW2": dW2,
            "db2": db2,
            "d_kernels1": d_kernels1,
            "d_biases1": d_biases1,
            "d_kernels2": d_kernels2,
            "d_biases2": d_biases2
        }

    # ==========================================
    # 8. 单样本计算 Loss 和梯度
    # ==========================================

    def train_one(self, image, label):

        probabilities, cache = self.forward(image)

        loss = -np.log(probabilities[label] + 1e-12)

        grads = self.backward(probabilities, label, cache)

        return loss, grads

    # ==========================================
    # 9. Mini-batch 训练
    # ==========================================

    def train_batch(self, X_batch, y_batch):

        total_grads = {
            "dW1": np.zeros_like(self.W1),
            "db1": np.zeros_like(self.b1),
            "dW2": np.zeros_like(self.W2),
            "db2": np.zeros_like(self.b2),
            "d_kernels1": np.zeros_like(self.kernels1),
            "d_biases1": np.zeros_like(self.biases1),
            "d_kernels2": np.zeros_like(self.kernels2),
            "d_biases2": np.zeros_like(self.biases2)
        }

        total_loss = 0.0

        # 逐张图片计算梯度
        for image, label in zip(X_batch, y_batch):

            loss, grads = self.train_one(image, label)

            total_loss += loss

            for key in total_grads:
                total_grads[key] += grads[key]

        # 平均梯度
        batch_size = len(X_batch)

        for key in total_grads:
            total_grads[key] /= batch_size

        # 更新参数
        self.W1 -= self.learning_rate * total_grads["dW1"]
        self.b1 -= self.learning_rate * total_grads["db1"]

        self.W2 -= self.learning_rate * total_grads["dW2"]
        self.b2 -= self.learning_rate * total_grads["db2"]

        self.kernels1 -= (
            self.learning_rate * total_grads["d_kernels1"]
        )
        self.biases1 -= (
            self.learning_rate * total_grads["d_biases1"]
        )

        self.kernels2 -= (
            self.learning_rate * total_grads["d_kernels2"]
        )
        self.biases2 -= (
            self.learning_rate * total_grads["d_biases2"]
        )

        return total_loss / batch_size

    # ==========================================
    # 10. fit：完整训练入口
    # ==========================================

    def fit(self, X, y):

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)

        if X.ndim == 3:
            X = X[..., np.newaxis]

        if X.ndim != 4 or X.shape[1:] != (28, 28, 1):
            raise ValueError(
                "X 的形状必须是 (N, 28, 28, 1) "
                "或 (N, 28, 28)"
            )

        if len(X) != len(y):
            raise ValueError("X 和 y 的样本数量不一致")

        for epoch in range(self.epochs):

            total_loss = 0.0

            # 打乱训练数据
            indices = self.rng.permutation(len(X))

            X_shuffled = X[indices]
            y_shuffled = y[indices]

            # Mini-batch
            for start in range(0, len(X), self.batch_size):

                end = start + self.batch_size

                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]

                loss = self.train_batch(X_batch, y_batch)

                total_loss += loss * len(X_batch)

            epoch_loss = total_loss / len(X)

            print(
                f"Epoch {epoch + 1}/{self.epochs}, "
                f"Loss: {epoch_loss:.4f}"
            )

        return self

    # ==========================================
    # 11. predict：预测
    # ==========================================

    def predict(self, image):

        image = np.asarray(image, dtype=np.float32)

        if image.shape == (28, 28):
            image = image[..., np.newaxis]

        if image.shape != (28, 28, 1):
            raise ValueError(
                "图片形状必须是 (28, 28) 或 (28, 28, 1)"
            )

        probabilities, _ = self.forward(image)

        prediction = int(np.argmax(probabilities))
        confidence = float(np.max(probabilities))

        return prediction, confidence

    # ==========================================
    # 12. save：保存模型
    # ==========================================

    def save(self):

        self.model_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        np.savez(
            self.model_path,
            kernels1=self.kernels1,
            biases1=self.biases1,
            kernels2=self.kernels2,
            biases2=self.biases2,
            W1=self.W1,
            b1=self.b1,
            W2=self.W2,
            b2=self.b2
        )

        print(f"CNN 模型已保存: {self.model_path}")

    # ==========================================
    # 13. load：加载模型
    # ==========================================

    def load(self):

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"找不到 CNN 模型文件: {self.model_path}"
            )

        data = np.load(self.model_path)

        self.kernels1 = data["kernels1"]
        self.biases1 = data["biases1"]

        self.kernels2 = data["kernels2"]
        self.biases2 = data["biases2"]

        self.W1 = data["W1"]
        self.b1 = data["b1"]

        self.W2 = data["W2"]
        self.b2 = data["b2"]

        data.close()

        print(f"CNN 模型已加载: {self.model_path}")

        return self
