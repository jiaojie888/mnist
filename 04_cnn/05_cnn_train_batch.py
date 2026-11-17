import numpy as np
import tensorflow as tf

# ==========================================
# 1. 加载 MNIST 数据
# ==========================================

(X_train, y_train), (X_test, y_test) = (
    tf.keras.datasets.mnist.load_data()
)

X_train = X_train.astype(np.float32) / 255.0
X_test = X_test.astype(np.float32) / 255.0

# 增加通道维度
X_train = X_train[:, :, :, np.newaxis]
X_test = X_test[:, :, :, np.newaxis]

print("训练集:", X_train.shape)
print("测试集:", X_test.shape)

# ==========================================
# 2. 卷积前向传播
# image: (H, W, C_in)
# kernels: (N, KH, KW, C_in)
# output: (OH, OW, N)
# ==========================================

def conv2d(image, kernels, biases):
    image_h, image_w, in_channels = image.shape

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

def conv2d_backward(image, kernels, d_output):
    image_h, image_w, in_channels = image.shape

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
# 保存最大值所在位置，供反向传播使用
# ==========================================

def max_pool2d_forward(feature_map, pool_size=2, stride=2):
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
# 6. 初始化 CNN 参数
# ==========================================

rng = np.random.default_rng(42)

# Conv1: 1 -> 8
kernels1 = (
    rng.standard_normal((8, 3, 3, 1)).astype(np.float32)
    * 0.1
)
biases1 = np.zeros(8, dtype=np.float32)

# Conv2: 8 -> 16
kernels2 = (
    rng.standard_normal((16, 3, 3, 8)).astype(np.float32)
    * 0.1
)
biases2 = np.zeros(16, dtype=np.float32)

# Dense1: 400 -> 128
W1 = (
    rng.standard_normal((400, 128)).astype(np.float32)
    * np.sqrt(2.0 / 400)
)
b1 = np.zeros(128, dtype=np.float32)

# Dense2: 128 -> 10
W2 = (
    rng.standard_normal((128, 10)).astype(np.float32)
    * np.sqrt(2.0 / 128)
)
b2 = np.zeros(10, dtype=np.float32)

# ==========================================
# 7. 前向传播
# ==========================================

def forward(image):
    cache = {}

    # Conv1 + ReLU + Pool1
    z1 = conv2d(image, kernels1, biases1)
    a1 = np.maximum(0, z1)

    pool1, pool1_indices = max_pool2d_forward(a1)

    # Conv2 + ReLU + Pool2
    z2 = conv2d(pool1, kernels2, biases2)
    a2 = np.maximum(0, z2)

    pool2, pool2_indices = max_pool2d_forward(a2)

    # Flatten
    flattened = pool2.reshape(-1)

    # Dense1 + ReLU
    z3 = flattened @ W1 + b1
    hidden = np.maximum(0, z3)

    # Dense2
    logits = hidden @ W2 + b2

    # Softmax
    exp_logits = np.exp(logits - np.max(logits))
    probabilities = exp_logits / np.sum(exp_logits)

    # 保存反向传播需要的数据
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
# 8. 完整反向传播
# ==========================================

def backward(probabilities, label, cache):

    # Softmax + Cross Entropy
    Y = np.zeros(10, dtype=np.float32)
    Y[label] = 1.0

    dz4 = probabilities - Y

    # Dense2
    dW2 = np.outer(cache["hidden"], dz4)
    db2 = dz4

    # Dense1
    dhidden = dz4 @ W2.T

    dz3 = dhidden * (cache["z3"] > 0)

    dW1 = np.outer(cache["flattened"], dz3)
    db1 = dz3

    dflattened = dz3 @ W1.T

    dpool2 = dflattened.reshape(cache["pool2"].shape)

    # Pool2 backward
    da2 = max_pool2d_backward(
        dpool2,
        cache["pool2_indices"],
        cache["z2"].shape
    )

    # ReLU2 backward
    dz2 = da2 * (cache["z2"] > 0)

    # Conv2 backward
    dpool1, d_kernels2, d_biases2 = conv2d_backward(
        cache["pool1"],
        kernels2,
        dz2
    )

    # Pool1 backward
    da1 = max_pool2d_backward(
        dpool1,
        cache["pool1_indices"],
        cache["z1"].shape
    )

    # ReLU1 backward
    dz1 = da1 * (cache["z1"] > 0)

    # Conv1 backward
    dimage, d_kernels1, d_biases1 = conv2d_backward(
        cache["image"],
        kernels1,
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
# 9. 单样本训练
# ==========================================

def train_one(image, label, learning_rate=0.001):

    # Forward
    probabilities, cache = forward(image)

    # Loss
    loss = -np.log(probabilities[label] + 1e-12)

    # Backward
    grads = backward(probabilities, label, cache)

    return loss, grads

# ==========================================
# 9.1. batch样本训练
# ==========================================

def train_batch(X_batch, y_batch, learning_rate=0.001):

    # 初始化梯度累加器
    total_grads = {
        "dW1": np.zeros_like(W1),
        "db1": np.zeros_like(b1),
        "dW2": np.zeros_like(W2),
        "db2": np.zeros_like(b2),
        "d_kernels1": np.zeros_like(kernels1),
        "d_biases1": np.zeros_like(biases1),
        "d_kernels2": np.zeros_like(kernels2),
        "d_biases2": np.zeros_like(biases2)
    }

    total_loss = 0

    # 逐张图片计算梯度
    for image, label in zip(X_batch, y_batch):

        loss, grads = train_one(image, label)

        total_loss += loss

        # 累加梯度
        for key in total_grads:
            total_grads[key] += grads[key]

    # 求平均梯度
    batch_size = len(X_batch)

    for key in total_grads:
        total_grads[key] /= batch_size

    # 更新参数
    W1[:] -= learning_rate * total_grads["dW1"]
    b1[:] -= learning_rate * total_grads["db1"]

    W2[:] -= learning_rate * total_grads["dW2"]
    b2[:] -= learning_rate * total_grads["db2"]

    kernels1[:] -= learning_rate * total_grads["d_kernels1"]
    biases1[:] -= learning_rate * total_grads["d_biases1"]

    kernels2[:] -= learning_rate * total_grads["d_kernels2"]
    biases2[:] -= learning_rate * total_grads["d_biases2"]

    return total_loss / batch_size

# ==========================================
# 10. 训练
# ==========================================

# 先使用5000张图片
X_train = X_train[:5000]
y_train = y_train[:5000]

batch_size = 32
epochs = 5
learning_rate = 0.01

for epoch in range(epochs):

    total_loss = 0.0

    indices = rng.permutation(len(X_train))

    X_shuffled = X_train[indices]
    y_shuffled = y_train[indices]

    for start in range(0, len(X_train), batch_size):

        end = start + batch_size

        X_batch = X_shuffled[start:end]
        y_batch = y_shuffled[start:end]

        loss = train_batch(
            X_batch,
            y_batch,
            learning_rate
        )

        total_loss += loss * len(X_batch)

    epoch_loss = total_loss / len(X_train)

    print(
        f"Epoch {epoch + 1}/{epochs}, "
        f"Loss: {epoch_loss:.4f}"
    )

# ==========================================
# 11. 测试
# ==========================================

def predict(image):
    probabilities, _ = forward(image)

    prediction = int(np.argmax(probabilities))
    confidence = float(np.max(probabilities))

    return prediction, confidence


test_size = 1000

correct = 0

for i in range(test_size):
    image = X_test[i]

    prediction, confidence = predict(image)

    if prediction == y_test[i]:
        correct += 1

accuracy = correct / test_size

print(f"测试数量: {test_size}")
print(f"测试准确率: {accuracy:.2%}")
print(f"错误数量: {test_size - correct}")


# ==========================================
# 12. 保存模型
# ==========================================

np.savez(
    "cnn_model.npz",
    kernels1=kernels1,
    biases1=biases1,
    kernels2=kernels2,
    biases2=biases2,
    W1=W1,
    b1=b1,
    W2=W2,
    b2=b2
)

print("模型已保存: cnn_model.npz")