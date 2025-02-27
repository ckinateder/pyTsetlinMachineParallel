from pyTsetlinMachineParallel.tm import MultiClassTsetlinMachine
import numpy as np
from pickle import dump, load
from time import time
import random
from keras.datasets import mnist
import matplotlib.pyplot as plt
import os

def visualize_activation_maps(teacher_model, student_model, distilled_model, sample, image_shape, output_filepath, class_idx=None):
    """
    Creates a visualization of activation maps for teacher, student, and distilled models.
    
    Args:
        teacher_model: The teacher TsetlinMachine model
        student_model: The student TsetlinMachine model
        distilled_model: The distilled TsetlinMachine model
        sample: Input sample to visualize
        image_shape: Tuple with image dimensions (height, width)
        output_filepath: Path where to save the output image
        class_idx: Specific class index to visualize. If None, uses the predicted class.
    """
    # Create figure with 1x4 layout
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    # If class_idx is not provided, use the predicted class from the teacher model
    if class_idx is None:
        # Get prediction from teacher model
        teacher_output = teacher_model.predict(sample.reshape(1, -1))
        class_idx = teacher_output[0]

    class_idx = int(class_idx)

    # Display original image
    axes[0].imshow(sample.reshape(image_shape), cmap='gray')
    axes[0].set_title("Original Sample")
    axes[0].axis('off')
    
    # Generate and display teacher activation map
    teacher_activation = teacher_model.get_activation_map(sample, class_idx=class_idx, image_shape=image_shape)
    axes[1].imshow(teacher_activation)
    axes[1].set_title("Teacher Model Features")
    axes[1].axis('off')
    
    # Generate and display student activation map
    student_activation = student_model.get_activation_map(sample, class_idx=class_idx, image_shape=image_shape)
    print(student_model.number_of_features)
    axes[2].imshow(student_activation)
    axes[2].set_title("Student Model Features")
    axes[2].axis('off')
    
    # Generate and display distilled activation map
    distilled_activation = distilled_model.get_activation_map(sample, class_idx=class_idx, image_shape=image_shape)
    print(distilled_model.number_of_features)
    axes[3].imshow(distilled_activation)
    axes[3].set_title("Distilled Model Features")
    axes[3].axis('off')
    
    # Add overall title
    plt.suptitle(f"Activation Maps Comparison for Class {class_idx}", fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    
    # Save figure
    #os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
    plt.savefig(output_filepath, dpi=150)
    plt.close()
    
    print(f"Activation maps comparison saved to {output_filepath}")

if __name__ == "__main__":

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
