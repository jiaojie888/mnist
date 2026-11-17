
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


# ==========================================
# 1. 加载测试图片和训练好的模型
# ==========================================

(_, _), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

X_test = X_test.astype(np.float32) / 255.0
X_test = X_test[:, :, :, np.newaxis]

model = np.load("cnn_model.npz")

kernels1 = model["kernels1"]
biases1 = model["biases1"]

kernels2 = model["kernels2"]
biases2 = model["biases2"]


# ==========================================
# 2. 卷积
# ==========================================

def conv2d(image, kernels, biases):

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
# 3. Max Pooling
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

                h = i * stride
                w = j * stride

                region = feature_map[
                    h:h + pool_size,
                    w:w + pool_size,
                    c
                ]

                output[i, j, c] = np.max(region)

    return output


# ==========================================
# 4. 选择一张图片
# ==========================================

image_index = 0

image = X_test[image_index]

print("真实数字:", y_test[image_index])


# ==========================================
# 5. 第一层卷积
# ==========================================

z1 = conv2d(image, kernels1, biases1)

# ReLU
a1 = np.maximum(0, z1)

# Pooling
pool1 = max_pool2d(a1)


# ==========================================
# 6. 第二层卷积
# ==========================================

z2 = conv2d(pool1, kernels2, biases2)

# ReLU
a2 = np.maximum(0, z2)

# Pooling
pool2 = max_pool2d(a2)


# ==========================================
# 7. 可视化第一层卷积核
# ==========================================

fig, axes = plt.subplots(2, 4, figsize=(10, 5))

fig.suptitle("First Layer: 8 Learned Kernels")

for i, ax in enumerate(axes.flat):

    kernel = kernels1[i, :, :, 0]

    ax.imshow(kernel, cmap="coolwarm")
    ax.set_title(f"Kernel {i}")
    ax.axis("off")

plt.tight_layout()
plt.show()


# ==========================================
# 8. 可视化第一层特征图
# ==========================================

fig, axes = plt.subplots(2, 4, figsize=(10, 5))

fig.suptitle("First Layer: Feature Maps")

for i, ax in enumerate(axes.flat):

    ax.imshow(a1[:, :, i], cmap="gray")
    ax.set_title(f"Feature {i}")
    ax.axis("off")

plt.tight_layout()
plt.show()


# ==========================================
# 9. 可视化第二层特征图
# ==========================================

fig, axes = plt.subplots(4, 4, figsize=(10, 10))

fig.suptitle("Second Layer: Feature Maps")

for i, ax in enumerate(axes.flat):

    ax.imshow(a2[:, :, i], cmap="gray")
    ax.set_title(f"Feature {i}")
    ax.axis("off")

plt.tight_layout()
plt.show()


# ==========================================
# 10. 查看原图和池化后的结果
# ==========================================

fig, axes = plt.subplots(1, 3, figsize=(10, 3))

axes[0].imshow(image[:, :, 0], cmap="gray")
axes[0].set_title("Original 28x28")

axes[1].imshow(pool1[:, :, 0], cmap="gray")
axes[1].set_title("Pool1 Feature 0")

axes[2].imshow(pool2[:, :, 0], cmap="gray")
axes[2].set_title("Pool2 Feature 0")

for ax in axes:
    ax.axis("off")

plt.tight_layout()
plt.show()
