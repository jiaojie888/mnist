import numpy as np
import tensorflow as tf

# Load the dataset
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

# Preprocess the data
X_train = X_train.reshape(-1, 28 * 28).astype('float32') / 255.0
X_test = X_test.reshape(-1, 28 * 28).astype('float32') / 255.0

print("Training data shape:", X_train.shape)
print("Training labels shape:", y_train.shape)
print("Test data shape:", X_test.shape)
print("Test labels shape:", y_test.shape)

# Initialize weights and bias for logistic regression
W = np.zeros((X_train.shape[1], 10))  # Initialize weights for 10 classes
b = np.zeros((10,))  # Initialize bias for 10 classes
print("Weights shape:", W.shape)
print("Bias shape:", b.shape)


x = X_train[0]  # Example input
z = x @ W + b  # Linear combination
print("True label:", y_train[0])
print("Linear combination (z):", z)
print("Linear combination (z) shape:", z.shape)

def softmax(z):
    exp_z = np.exp(z - np.max(z))  # Subtract max for numerical stability
    return exp_z / np.sum(exp_z)

probabilities = softmax(z)
print("Predicted probabilities:", probabilities) 
print("Sum of probabilities", np.sum(probabilities))  # Should sum to 1
print("Predicted class:", np.argmax(probabilities))  # Class with highest probability

def cross_entropy_loss(probabilities, y):
    return -np.log(probabilities[y] + 1e-15)  # Add small value to avoid log(0)

loss = cross_entropy_loss(probabilities, y_train[0])


def train_step(X_batch, y_batch, W, b, learning_rate=0.01):
    batch_size = X_batch.shape[0]

    # 1. Forward pass
    z = X_batch @ W + b

    # 2. Compute probabilities using softmax
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))  # Subtract max for numerical stability
    probabilities = exp_z / np.sum(exp_z, axis=1, keepdims=True)

    # 3. Compute loss
    loss = np.mean([-np.log(probabilities[np.arange(batch_size), y_batch] + 1e-15)])

    # 4. Backward pass
    dz = probabilities.copy()
    dz[np.arange(batch_size), y_batch] -= 1  # Gradient of loss w.r.t z
    dz /= batch_size

    dW = X_batch.T @ dz  # Gradient of loss w.r.t W
    db = np.sum(dz, axis=0)  # Gradient of loss w.r.t b

    # Update weights and bias
    W -= learning_rate * dW
    b -= learning_rate * db

    return loss

# Example training step
batch_size = 128
learning_rate = 0.01
epochs = 10

# Re-initialize weights and bias for training
W = np.zeros((X_train.shape[1], 10))  # Re-initialize weights for training
b = np.zeros((10,))  # Re-initialize bias for training

for epoch in range(epochs):
    # Shuffle the training data
    indices = np.arange(X_train.shape[0])
    np.random.shuffle(indices)
    X_train_shuffled = X_train[indices]
    y_train_shuffled = y_train[indices]

    total_loss = 0

    for i in range(0, X_train.shape[0], batch_size):
        X_batch = X_train_shuffled[i:i + batch_size]
        y_batch = y_train_shuffled[i:i + batch_size]

        loss = train_step(X_batch, y_batch, W, b, learning_rate)
        total_loss += loss

    average_loss = total_loss / (X_train.shape[0] // batch_size)
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {average_loss:.4f}")


def predict(X, W, b):
    z = X @ W + b
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))  # Subtract max for numerical stability
    probabilities = exp_z / np.sum(exp_z, axis=1, keepdims=True)
    return np.argmax(probabilities, axis=1)

predictions = predict(X_test, W, b)
accuracy = np.mean(predictions == y_test)
wrong_predictions = np.sum(predictions != y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")
print(f"Number of wrong predictions: {wrong_predictions}")
