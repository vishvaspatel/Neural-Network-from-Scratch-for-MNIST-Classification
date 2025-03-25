# Neural Network from Scratch for MNIST Classification

## Introduction
This project involves implementing a feedforward neural network from scratch using Python to classify the MNIST dataset into 10 classes (digits 0-9). The model is built without relying on high-level deep learning libraries for architecture and training, focusing on implementing forward propagation, backpropagation, and optimization manually.

## Objective
The key objectives of this project include:
- Implementing a feedforward neural network with at least two fully connected layers.
- Training and evaluating the network using randomized train-test splits (70:30, 80:20, or 90:10).
- Initializing weights randomly using a specified seed (based on Roll Number) and setting biases to 1.
- Training the model for 25 epochs and analyzing performance through:
  - Accuracy and loss curves.
  - Confusion matrices for validation and test sets.
- Exploring different activation functions (ReLU and ELU) and modern weight initialization strategies.
- Reporting the total trainable parameters.

## Dataset
The MNIST dataset consists of 70,000 grayscale images of handwritten digits (0-9), each of size 28×28 pixels. The dataset is split into:
- **Training set:** 60,000 images
- **Test set:** 10,000 images

### Preprocessing Steps:
1. **Normalization:**
   - Pixel values are scaled to [0,1] for better convergence.
2. **Flattening:**
   - Each image is reshaped into a 784-dimensional vector.
3. **One-hot Encoding:**
   - Labels are converted into one-hot vectors.
4. **Randomized Train-Test Split:**
   - The dataset is split into training and validation sets in a ratio of 70:30, 80:20, or 90:10.

## Neural Network Architecture
The neural network consists of the following layers:
1. **Input Layer:** 784 neurons (one per pixel).
2. **Hidden Layers:**
   - First hidden layer: 128 neurons.
   - Second hidden layer: 64 neurons.
   - Activation functions: **ReLU** and **ELU** (compared in experiments).
3. **Output Layer:**
   - 10 neurons (one per class) with softmax activation.

### Initialization Strategy:
- Weights initialized using **He Initialization** to prevent vanishing/exploding gradients.
- Bias values initialized to **1**.

## Implementation Details
### Forward Propagation:
- Inputs pass through each layer using weight and bias matrices.
- Activations computed using **ReLU** or **ELU** for hidden layers and **softmax** for output.

### Loss Function:
- **Cross-Entropy Loss** used to measure prediction error.

### Backward Propagation:
- Gradients computed using the chain rule.
- Backpropagation applied through each layer.

### Parameter Updates:
- Weights and biases updated using **Stochastic Gradient Descent (SGD)**.

### Regularization:
- **Dropout** initially tested but removed in final implementation due to minimal overfitting.

## Evaluation Metrics
1. **Accuracy and Loss Curves:**
   - Tracked training and validation accuracy/loss for 25 epochs.
2. **Confusion Matrices:**
   - Generated for validation and test sets to analyze class-wise performance.

## Experimental Configurations
1. **Activation Functions Compared:**
   - **ReLU**: Efficient but may suffer from dead neurons.
   - **ELU**: Handles small gradients better.
2. **Train-Test Splits:**
   - Experiments performed with **70:30, 80:20, and 90:10** train-validation splits.
3. **Total Trainable Parameters:**
   - The network consists of **109,386** trainable parameters.

## Results
| Activation Function | Final Test Accuracy |
|---------------------|--------------------|
| **ReLU**           | **97.22%**         |
| **ELU**            | **97.18%**         |

### Observations
- **High accuracy** achieved for both activation functions.
- **Minimal overfitting** observed, making dropout unnecessary.
- **ReLU performed slightly better** than ELU in terms of test accuracy.
- **Confusion matrices** showed minor misclassification among visually similar digits.
- **He Initialization** ensured smooth training without gradient vanishing/explosion.

## Conclusion
This project successfully implemented a **neural network from scratch** for MNIST classification, achieving over **97% accuracy**. The study compared **ReLU vs. ELU**, experimented with **train-test splits**, and analyzed **confusion matrices**. The results highlight the effectiveness of manual NN implementation without deep learning libraries.

