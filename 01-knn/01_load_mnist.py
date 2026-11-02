import tensorflow as tf
import matplotlib.pyplot as plt

# 1. 加载 MNIST 数据集
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

# 2. 查看数据形状
print("训练集图片：", X_train.shape)
print("训练集标签：", y_train.shape)

print("测试集图片：", X_test.shape)
print("测试集标签：", y_test.shape)

# 3. 显示第一张图片
number_of_images = 10
plt.imshow(X_train[number_of_images], cmap="gray")
plt.title(f"Label: {y_train[number_of_images]}")
plt.axis("off")
plt.show()



# 查看第一张图片的像素矩阵
print("图片形状：", X_train[number_of_images].shape)

print("像素矩阵：")
print(X_train[number_of_images])

# 查看像素范围
print("最小像素值：", X_train[number_of_images].min())
print("最大像素值：", X_train[number_of_images].max())


# 将图片展平
image = X_train[number_of_images].flatten()

print("展平后的形状：", image.shape)

print("前20个像素：")
print(image[:2000])


plt.show()

plt.hist(image, bins=30)
plt.title("Pixel Value Distribution")
plt.xlabel("Pixel Value")
plt.ylabel("Number of Pixels")
plt.show()


# 将像素数据转换为 float32 类型，并归一化到 0～1
X_train_normalized = X_train.astype("float32") / 255.0
X_test_normalized = X_test.astype("float32") / 255.0

# 查看归一化前后的数据
print("归一化前：")
print(X_train[number_of_images].min(), X_train[number_of_images].max())

print("归一化后：")
print(X_train_normalized[number_of_images].min(), X_train_normalized[number_of_images].max())

# 查看数据类型
print("归一化前的数据类型：", X_train.dtype)
print("归一化后的数据类型：", X_train_normalized.dtype)


x_train_flattened = X_train_normalized.reshape(X_train_normalized.shape[0], -1)
x_test_flattened = X_test_normalized.reshape(X_test_normalized.shape[0], -1)

# 查看展平后的数据形状
print("展平后的训练集形状：", x_train_flattened.shape)
print("展平后的测试集形状：", x_test_flattened.shape)

print("训练集标签形状：", y_train.shape)
print("测试集标签形状：", y_test.shape)