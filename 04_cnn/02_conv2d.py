import numpy as np

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



image = np.ones((5, 5, 3), dtype=np.float32)
kernels = np.ones((2, 3, 3, 3), dtype=np.float32)
biases = np.zeros(2, dtype=np.float32)

feature_maps = conv2d(image=image, kernels=kernels, biases=biases)
feature_maps = np.maximum(0, feature_maps)
pooled = max_pool2d(feature_maps)

print("feature_maps shape", feature_maps.shape)
print(feature_maps[:, :, 0])
print("pooled shape", pooled.shape)
print(pooled[:, :, 0])
