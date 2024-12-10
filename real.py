import numpy as np
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


# 1. Load and normalize the MNIST dataset
def load_data():
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Lambda(lambda x: x.view(-1))])

    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    x_train = train_dataset.data.numpy().reshape(-1, 784) / 255.0
    x_test = test_dataset.data.numpy().reshape(-1, 784) / 255.0
    return x_train, x_test


# Load data
x_train, x_test = load_data()

# 2. Initialize weights and biases
W1 = np.random.randn(512, 784) * 0.01
b1 = np.zeros((512, 1))
W2 = np.random.randn(256, 512) * 0.01
b2 = np.zeros((256, 1))
W3 = np.random.randn(128, 256) * 0.01
b3 = np.zeros((128, 1))
W4 = np.random.randn(256, 128) * 0.01
b4 = np.zeros((256, 1))
W5 = np.random.randn(512, 256) * 0.01
b5 = np.zeros((512, 1))
W6 = np.random.randn(784, 512) * 0.01
b6 = np.zeros((784, 1))


# 3. Define activation functions
def relu(x):
    return np.maximum(0, x)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# 4. Implement forward pass (Encoder + Decoder)
def forward(x):
    # Encoder
    h1 = relu(np.dot(W1, x) + b1)
    h2 = relu(np.dot(W2, h1) + b2)
    z = relu(np.dot(W3, h2) + b3)  # Latent representation

    # Decoder
    h3 = relu(np.dot(W4, z) + b4)
    h4 = relu(np.dot(W5, h3) + b5)
    x_reconstructed = sigmoid(np.dot(W6, h4) + b6)

    return z, x_reconstructed


# 5. Compute Mean Squared Error Loss
def mse_loss(x, x_reconstructed):
    return np.mean((x - x_reconstructed) ** 2)


# 6. Implement backpropagation and weight updates
def backward(x, h1, h2, z, h3, h4, x_reconstructed, lr=0.001):
    global W1, b1, W2, b2, W3, b3, W4, b4, W5, b5, W6, b6

    # Output layer gradient
    output_grad = -(x - x_reconstructed) * (x_reconstructed * (1 - x_reconstructed))

    # Decoder gradients
    dW6 = np.dot(output_grad, h4.T)
    db6 = np.sum(output_grad, axis=1, keepdims=True)

    dh4 = np.dot(W6.T, output_grad) * (h4 > 0)
    dW5 = np.dot(dh4, h3.T)
    db5 = np.sum(dh4, axis=1, keepdims=True)

    dh3 = np.dot(W5.T, dh4) * (h3 > 0)
    dW4 = np.dot(dh3, z.T)
    db4 = np.sum(dh3, axis=1, keepdims=True)

    # Encoder gradients
    dz = np.dot(W4.T, dh3) * (z > 0)
    dW3 = np.dot(dz, h2.T)
    db3 = np.sum(dz, axis=1, keepdims=True)

    dh2 = np.dot(W3.T, dz) * (h2 > 0)
    dW2 = np.dot(dh2, h1.T)
    db2 = np.sum(dh2, axis=1, keepdims=True)

    dh1 = np.dot(W2.T, dh2) * (h1 > 0)
    dW1 = np.dot(dh1, x.T)
    db1 = np.sum(dh1, axis=1, keepdims=True)

    # Update weights and biases
    W1 -= lr * dW1
    b1 -= lr * db1
    W2 -= lr * dW2
    b2 -= lr * db2
    W3 -= lr * dW3
    b3 -= lr * db3
    W4 -= lr * dW4
    b4 -= lr * db4
    W5 -= lr * dW5
    b5 -= lr * db5
    W6 -= lr * dW6
    b6 -= lr * db6


# 7. Implement training loop with mini-batch
def train(x_train, epochs=100, lr=0.001, batch_size=64):
    x_train = x_train.T
    losses = []
    latent_codes = []
    reconstructed_data = []

    for epoch in range(epochs):
        epoch_loss = 0
        # Shuffle the dataset
        indices = np.arange(x_train.shape[1])
        np.random.shuffle(indices)
        x_train = x_train[:, indices]

        # Process mini-batches
        for i in range(0, x_train.shape[1], batch_size):
            x_batch = x_train[:, i:i + batch_size]  # Get a batch of samples, shape (784, batch_size)
            h1 = relu(np.dot(W1, x_batch) + b1)
            h2 = relu(np.dot(W2, h1) + b2)
            z = relu(np.dot(W3, h2) + b3)
            h3 = relu(np.dot(W4, z) + b4)
            h4 = relu(np.dot(W5, h3) + b5)
            x_reconstructed = sigmoid(np.dot(W6, h4) + b6)

            # Compute loss and accumulate epoch loss
            loss = mse_loss(x_batch, x_reconstructed)
            epoch_loss += loss * batch_size  # Scale the loss by batch size

            # Backward pass for each batch
            backward(x_batch, h1, h2, z, h3, h4, x_reconstructed, lr)

        avg_loss = epoch_loss / x_train.shape[1]
        losses.append(avg_loss)
        print(f'Epoch {epoch + 1}, Loss: {avg_loss:.4f}')
    # Collect results for visualization

    for i in range(x_train.shape[1]):
        x = x_train[:, i:i + 1]
        z, x_reconstructed = forward(x)
        latent_codes.append(z.flatten())
        reconstructed_data.append(x_reconstructed.flatten())

    return losses, np.array(latent_codes), np.array(reconstructed_data)


# 8. Plot the loss curve
def plot_loss_curve(losses):
    plt.figure(figsize=(8, 6))
    plt.plot(losses, label='MSE Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()
    plt.show()


# 9. Visualize latent codes using PCA
def visualize_latent_space(latent_codes):
    pca = PCA(n_components=2)
    reduced_codes = pca.fit_transform(latent_codes)
    plt.figure(figsize=(8, 6))
    plt.scatter(reduced_codes[:, 0], reduced_codes[:, 1], s=5, alpha=0.6)
    plt.title('Latent Space Visualization (PCA)')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.show()


# 10. Show original and reconstructed images side by side
def show_images(original, reconstructed, n=10):
    plt.figure(figsize=(20, 4))
    for i in range(n):
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(original[i].reshape(28, 28), cmap='gray')
        plt.axis('off')

        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(reconstructed[i].reshape(28, 28), cmap='gray')
        plt.axis('off')
    plt.show()


# Call train function
losses, latent_codes, reconstructed_data = train(x_train, epochs=100, lr=0.001)

# 8. Plot the loss curve
plot_loss_curve(losses)

# 9. Visualize latent codes using PCA
visualize_latent_space(latent_codes)

# 10. Show original and reconstructed images side by side
# Get the reconstructed images for the first 10 samples
original_samples = x_train[:10]
reconstructed_samples = [forward(x.reshape(-1, 1))[1].flatten() for x in original_samples]

# Display original and reconstructed images (ensure correct order)
show_images(original_samples, reconstructed_samples)
