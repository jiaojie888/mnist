
from pathlib import Path
import numpy as np
import tensorflow as tf

# 项目根目录
ROOT = Path(__file__).resolve().parents[1]

# 模型保存目录
MODEL_DIR = ROOT / "models" / "04_cnn"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "cnn_tensorflow.keras"


def load_data():
    (X_train, y_train), (X_test, y_test) = (
        tf.keras.datasets.mnist.load_data()
    )

    # 归一化
    X_train = X_train.astype("float32") / 255.0
    X_test = X_test.astype("float32") / 255.0

    # 增加通道维度
    X_train = X_train[..., np.newaxis]
    X_test = X_test[..., np.newaxis]

    return X_train, y_train, X_test, y_test


def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(28, 28, 1)),

        tf.keras.layers.Conv2D(
            8, kernel_size=3, activation="relu"
        ),
        tf.keras.layers.MaxPooling2D(pool_size=2),

        tf.keras.layers.Conv2D(
            16, kernel_size=3, activation="relu"
        ),
        tf.keras.layers.MaxPooling2D(pool_size=2),

        tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(128, activation="relu"),

        tf.keras.layers.Dense(10, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


def main():
    X_train, y_train, X_test, y_test = load_data()

    print("训练集:", X_train.shape)
    print("测试集:", X_test.shape)

    model = build_model()

    model.summary()

    # 完整训练集
    model.fit(
        X_train,
        y_train,
        epochs=5,
        batch_size=32,
        validation_split=0.1
    )

    # 完整测试集
    loss, accuracy = model.evaluate(
        X_test,
        y_test,
        verbose=1
    )

    print(f"测试 Loss: {loss:.4f}")
    print(f"测试准确率: {accuracy * 100:.2f}%")

    # 保存模型
    model.save(MODEL_PATH)

    print(f"模型已保存: {MODEL_PATH}")


if __name__ == "__main__":
    main()
