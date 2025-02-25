from pyTsetlinMachineParallel.tm import MultiClassTsetlinMachine
import numpy as np
from pickle import dump, load
from time import time
import random
from keras.datasets import mnist
import matplotlib.pyplot as plt

# Load MNIST dataset
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

# Binarize the data
X_train = np.where(X_train.reshape((X_train.shape[0], 28*28)) > 75, 1, 0) 
X_test = np.where(X_test.reshape((X_test.shape[0], 28*28)) > 75, 1, 0)

# Load the pre-trained model
tm = load(open("tm.pkl", "rb"))

# Select a random class
random_class = random.randint(0, 9)
print(f"Visualizing feature importance for digit class: {random_class}")

# Find indices of test samples belonging to the selected class
class_indices = np.where(Y_test == random_class)[0]

# Shuffle and take 16 random samples
random.shuffle(class_indices)
num_samples = min(16, len(class_indices))
selected_indices = class_indices[:num_samples]

# Create a figure with a 4x8 grid (each sample gets 2 plots - original and activation)
fig, axes = plt.subplots(4, 8, figsize=(16, 8))

# Process 16 samples
for i, idx in enumerate(selected_indices):
    if i >= 16:  # Safety check
        break
    
    # Calculate row and column indices for this sample pair
    row = i // 4
    col = (i % 4) * 2  # Each sample takes 2 columns
    
    # Get the sample
    test_example = X_test[idx]
    
    # Display original image on the left
    axes[row, col].imshow(test_example.reshape(28, 28), cmap='gray')
    axes[row, col].set_title(f"Original #{i+1}")
    axes[row, col].axis('off')
    
    # Generate and display activation map on the right
    activation_map = tm.get_activation_map(test_example, class_idx=random_class, image_shape=(28, 28))
    axes[row, col+1].imshow(activation_map)
    axes[row, col+1].set_title(f"Features #{i+1}")
    axes[row, col+1].axis('off')

# Handle any empty subplots
for i in range(num_samples, 16):
    row = i // 4
    col = (i % 4) * 2
    axes[row, col].axis('off')
    axes[row, col+1].axis('off')

plt.suptitle(f"Feature Importance for Digit Class {random_class}", fontsize=16)
plt.tight_layout()
plt.subplots_adjust(top=0.92)
plt.savefig(f'mnist_feature_importance_class_{random_class}.png', dpi=150)
plt.close()

print(f"Visualization saved as 'mnist_feature_importance_class_{random_class}.png'")
