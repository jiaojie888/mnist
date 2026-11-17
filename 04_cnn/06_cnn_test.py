
import numpy as np
import tensorflow as tf


# ==========================================
# 1. 加载 MNIST 测试数据
# ==========================================

(_, _), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

X_test = X_test.astype(np.float32) / 255.0

# 增加通道维度
X_test = X_test[:, :, :, np.newaxis]

print("测试集:", X_test.shape)


# ==========================================
# 2. 加载训练好的 CNN 模型
# ==========================================

model = np.load("cnn_model.npz")

kernels1 = model["kernels1"]
biases1 = model["biases1"]

kernels2 = model["kernels2"]
biases2 = model["biases2"]

W1 = model["W1"]
b1 = model["b1"]

W2 = model["W2"]
b2 = model["b2"]


# ==========================================
# 3. 卷积前向传播
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
# 4. Max Pooling
# ==========================================

def max_pool2d(feature_map, pool_size=2, stride=2):

    height, width, channels = feature_map.shape

    output_h = (height - pool_size) // stride + 1
    output_w = (width - pool_size) // stride + 1

    output = np.zeros(
        (output_h, output_w, channels),
        dtype=np.float32
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

                output[i, j, c] = np.max(region)

    return output


# ==========================================
# 5. CNN 前向传播
# ==========================================

def forward(image):

    # Conv1 + ReLU + Pool1
    z1 = conv2d(image, kernels1, biases1)
    a1 = np.maximum(0, z1)

    pool1 = max_pool2d(a1)

    # Conv2 + ReLU + Pool2
    z2 = conv2d(pool1, kernels2, biases2)
    a2 = np.maximum(0, z2)

    pool2 = max_pool2d(a2)

    # Flatten
    flattened = pool2.reshape(-1)

    # Dense1 + ReLU
    z3 = flattened @ W1 + b1
    hidden = np.maximum(0, z3)

    # Dense2 + Softmax
    logits = hidden @ W2 + b2

    exp_logits = np.exp(logits - np.max(logits))
    probabilities = exp_logits / np.sum(exp_logits)

    return probabilities


# ==========================================
# 6. 测试完整 10000 张图片
# ==========================================

correct = 0
total = len(X_test)

for i in range(total):

    probabilities = forward(X_test[i])

    prediction = np.argmax(probabilities)

    if prediction == y_test[i]:
        correct += 1

    if (i + 1) % 1000 == 0:
        print(f"已测试: {i + 1}/{total}")


accuracy = correct / total

print("\n========== 测试结果 ==========")
print(f"测试数量: {total}")
print(f"正确数量: {correct}")
print(f"错误数量: {total - correct}")
print(f"测试准确率: {accuracy:.2%}")
