# **Autoencoder for MNIST Dataset**

A custom-built autoencoder designed to learn latent representations and reconstruct MNIST handwritten digit images using NumPy and Python. This project demonstrates implementing a simple deep learning model from scratch, including forward propagation, backpropagation, and visualization of results.

---

## **Features**
- Implementation of a **6-layer neural network**:
  - **Encoder:** 784 → 512 → 256 → 128
  - **Decoder:** 128 → 256 → 512 → 784
- **Mini-batch gradient descent** for efficient training.
- Custom loss computation using Mean Squared Error (MSE).
- Training progress with a final **loss of 0.0040** after 100 epochs.

---

## **Setup Instructions**
### **Prerequisites**
Ensure you have the following installed:
- Python 3.6 or above
- NumPy
- matplotlib
- TorchVision
- scikit-learn

### **Installation**
Clone this repository and install the required dependencies:
```bash
git clone https://github.com/your-username/Autoencoder-MNIST.git
cd Autoencoder-MNIST
pip install -r requirements.txt
```

---

## **Usage**
### **Running the Code**
To execute the script, use the following command:
```bash
python mnist_autoencoder.py
```

---

## **Results**
### **Training Loss**
The training log is available in `training_log.txt` and shows the following progress:
```
Epoch 1, Loss: 0.0654
Epoch 2, Loss: 0.0505
...
Epoch 100, Loss: 0.0040
```

### **Visualizations**
- **Loss Curve, Latent Space, and Reconstructions:** Visualizations will be added in future updates. These will include:
  - A loss curve showing the training progress.
  - A PCA visualization of the latent space.
  - A comparison of original and reconstructed MNIST images.

---

## **Project Structure**
```plaintext
Autoencoder-MNIST/
│
├── mnist_autoencoder.py           # Main Python script
├── training_log.txt               # Training logs
├── README.md                      # Documentation
```

---

## **How It Works**
1. **Encoder:** Compresses the 784-dimensional input data (28x28 images) into a 128-dimensional latent representation.
2. **Decoder:** Reconstructs the original input data from the latent representation.
3. **Training:** Uses mini-batch gradient descent and Mean Squared Error (MSE) to optimize the weights and biases.

---

## **Technologies Used**
- **Python:** Core programming language for implementation.
- **NumPy:** Efficient matrix operations.
- **TorchVision:** For MNIST dataset preprocessing.
- **Matplotlib:** Visualization of results.
- **scikit-learn:** PCA for latent space analysis.

---

## **Credits**
This project was inspired by deep learning principles and built entirely from scratch as part of an effort to understand and visualize autoencoders.
