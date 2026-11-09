import time
import tensorflow as tf
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# 1. 加载 MNIST 数据集
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

# 2. 数据预处理
X_train = X_train.reshape(X_train.shape[0], -1) / 255.0  # 展平并归一化
X_test = X_test.reshape(X_test.shape[0], -1) / 255.0  # 展平并归一化

# 3. 创建 KNN 分类器
for k in [1, 3, 5, 10, 20]:
    knn = KNeighborsClassifier(n_neighbors=k, algorithm='auto', n_jobs=-1)

    # 4. 训练模型
    start_time = time.time()
    knn.fit(X_train, y_train)
    training_time = time.time() - start_time
    print(f"k={k} 训练时间: {training_time:.2f} 秒")

    # 5. 在测试集上进行预测
    start_time = time.time()
    y_pred = knn.predict(X_test)
    prediction_time = time.time() - start_time
    print(f"k={k} 预测时间: {prediction_time:.2f} 秒")

    # 6. 计算准确率
    accuracy = accuracy_score(y_test, y_pred)
    print(f"k={k} 准确率: {accuracy:.2f}")

    # 7. 显示前10个错误分类的样本
    import matplotlib.pyplot as plt

    wrong_indices = [i for i in range(len(y_test)) if y_test[i] != y_pred[i]]
    print(f"错误分类的样本数量: {len(wrong_indices)}\n")

    plt.figure(figsize=(12, 5))
    for i, idx in enumerate(wrong_indices[:10]):
        plt.subplot(2, 5, i + 1)
        plt.imshow(X_test[idx].reshape(28, 28), cmap='gray')
        plt.title(f"True: {y_test[idx]}, Pred: {y_pred[idx]}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()

