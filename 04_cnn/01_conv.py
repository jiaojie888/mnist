import numpy as np

def conv2d(image, kernel, bias=0):
    image_h, image_w = image.shape
    kernel_h, kernel_w = kernel.shape

    output_h = image_h - kernel_h + 1
    output_w = image_w - kernel_w + 1

    output = np.zeros((output_h, output_w))

    for i in range(output_h):
        for j in range(output_w):
            region = image[
                i:i + kernel_h,
                j:j + kernel_w
            ]

            output[i, j] = np.sum(region * kernel) + bias

    return output

def conv2d_multi(image, kernels, biases):
    num_filters, kernel_h, kernel_w = kernels.shape

    image_h, image_w = image.shape

    output_h = image_h - kernel_h + 1
    output_w = image_w - kernel_w + 1

    output = np.zeros((output_h, output_w, num_filters))

    for f in range(num_filters):
        kernel = kernels[f]
        bias = biases[f]

        for i in range(output_h):
            for j in range(output_w):
                region = image[
                    i:i + kernel_h,
                    j:j + kernel_w
                ]

                output[i, j, f] = (
                    np.sum(region * kernel) + bias
                )

    return output




image = np.array([
    [1, 1, 0, 0, 0],
    [1, 1, 0, 0, 0],
    [1, 1, 0, 0, 0],
    [1, 1, 0, 0, 0],
    [1, 1, 0, 0, 0]
])

kernel = np.array([
    [1, 0, -1],
    [1, 0, -1],
    [1, 0, -1]
])

kernels = np.array([
    [
        [1, 0, -1],
        [1, 0, -1],
        [1, 0, -1]
    ],
    [
        [1, 1, 1],
        [0, 0, 0],
        [-1, -1, -1]
    ],
    [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ]
])

biases = np.array([0, 0, 0])

feature_map = conv2d(image=image, kernel=kernel, bias=2)
feature_maps = conv2d_multi(image=image, kernels=kernels, biases=biases)

print(feature_map)
print("输出形状：", feature_maps.shape)
print("第一张特征图：")
print(feature_maps[:, :, 0])
print("第二张特征图：")
print(feature_maps[:, :, 1])
print("第三张特征图：")
print(feature_maps[:, :, 2])
