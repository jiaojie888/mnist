import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


def conv2d(image, kernels, biases):

    # input shape
    image_h, image_w, in_channels = image.shape

    # kernel shape
    num_filters, kernel_h, kernel_w, _ = kernels.shape

    # stride=1, padding=0
    output_h = image_h - kernel_h + 1
    output_w = image_w - kernel_w + 1

    output = np.zeros((output_h, output_w, num_filters), dtype=np.float32)

    # traversal num_filters
    for f in range(num_filters):

        # traversal i,j in output
        for i in range(output_h):
            for j in range(output_w):

                # 取出所有输入通道的局部区域
                region = image[
                    i:i + kernel_h,
                    j:j + kernel_w
                ]

                output[i, j, f] = (
                    np.sum(region * kernels[f]) + biases[f]
                )

    return output

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
                region = feature_map[
                    i * stride : i * stride + pool_size,
                    j * stride : j * stride + pool_size,
                    c
                ]

                output[i, j, c] = np.max(region)

    return output


# load dataset
(X_train, y_train), (X_test, y_test) = (
    tf.keras.datasets.mnist.load_data()
)

# pick first image
image = X_train[0].astype(np.float32) / 255.0

# (28, 28) -> (28, 28, 1)
image = image[:, :, np.newaxis]

print("gt:", y_train[0])
print("image shape:", image.shape)

plt.imshow(image[:, :, 0], cmap="gray")
plt.title(f"Label: {y_train[0]}")
plt.show()

# kernels
kernels = np.random.randn(8, 3, 3, 1).astype(np.float32) * 0.1
# biases
biases = np.zeros(8, dtype=np.float32)

feature_maps = conv2d(image, kernels, biases)
feature_maps = np.maximum(0, feature_maps)
pooled = max_pool2d(feature_maps)

print("输入形状：", image.shape)
print("卷积后：", feature_maps.shape)
print("池化后：", pooled.shape)


# 我们把原始图片和前8张特征图展示出来
fig, axes = plt.subplots(3, 3, figsize=(9, 9))
axes[0, 0].imshow(image[:, :, 0], cmap="gray")
axes[0, 0].set_title("Original")

for i in range(8):
    row = (i + 1) // 3
    col = (i + 1) % 3

    axes[row, col].imshow(feature_maps[:, :, 0], cmap="gray")
    axes[row, col].set_title(f"Feature Map{i + 1}")

for ax in axes.flat:
    ax.axis("off")

plt.tight_layout()
plt.show()

# 第二层：16个卷积核
kernels2 = (
    np.random.randn(16, 3, 3, 8).astype(np.float32) * 0.1
)

biases2 = np.zeros(16, dtype=np.float32)

# 第一层卷积
feature_maps1 = conv2d(image, kernels, biases)

# 第一层 ReLU
feature_maps1 = np.maximum(0, feature_maps1)

# 第一层池化
pooled1 = max_pool2d(feature_maps1)


# 第二层卷积
feature_maps2 = conv2d(pooled1, kernels2, biases2)

# 第二层 ReLU
feature_maps2 = np.maximum(0, feature_maps2)

# 第二层池化
pooled2 = max_pool2d(feature_maps2)


print("原始图片：", image.shape)
print("第一层卷积：", feature_maps1.shape)
print("第一层池化：", pooled1.shape)
print("第二层卷积：", feature_maps2.shape)
print("第二层池化：", pooled2.shape)