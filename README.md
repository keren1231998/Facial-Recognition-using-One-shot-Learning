# Facial-Recognition-using-One-shot-Learning
## Authors
- **Keren Gorelik**
- **Lior Kobi**

## Project Overview
This project is a deep learning assignment focused on facial recognition using one-shot learning with a Siamese Neural Network architecture. The objective is to enable the model to distinguish between images of the same or different individuals, even with minimal examples. Using convolutional neural networks (CNNs), we tested the model on the "Labeled Faces in the Wild" dataset.

## Dataset
- **Dataset Used**: Labeled Faces in the Wild (LFW-a)
- **Total Size**: 3200 images
  - Train Set: 2200 images (1100 per class)
  - Test Set: 1000 images (500 per class)

## Model Architecture
### Siamese Network with CNN Layers
- **Convolutional Layers**: 4 layers with increasing filter sizes and ReLU activations.
- **Fully Connected Layers**: 2 fully connected layers to project features to a similarity score.
- **Enhancements**:
  - **Batch Normalization**: Stabilizes training and improves model performance.
  - **Dropout**: Reduces overfitting in fully connected layers.
  - **Combination Model**: Integrates both batch normalization and dropout.

## Experiments on Hyperparameters
We conducted grid search over various hyperparameters to optimize performance, including:
- **Batch Size**: Tested values - [16, 32, 64]
- **Epochs**: 20, 25
- **Learning Rate**: 0.0001, 0.001
- **Loss Functions**: Binary Cross-Entropy with Logits, Contrastive Loss
- **Optimizers**: Adam, SGD

### Best Model Configuration
- **Batch Size**: 32
- **Learning Rate**: 0.0001
- **Optimizer**: Adam
- **Loss Function**: Binary Cross-Entropy Loss

## Results
### Model Comparisons
1. **Base Model (No Batch Norm or Dropout)**:
   - Train Accuracy: 82.16%
   - Validation Accuracy: 64.09%
2. **Model with Batch Normalization**:
   - Train Accuracy: 100%
   - Validation Accuracy: 69.55%
3. **Model with Dropout**:
   - Train Accuracy: 50.4%
   - Validation Accuracy: 51.59%
4. **Model with Batch Norm and Dropout**:
   - Train Accuracy: 73.47%
   - Validation Accuracy: 59.55%

The model with batch normalization achieved the best balance of accuracy and generalization with a validation accuracy of 69.55%.

## Testing
- **Test Accuracy**: 70.26%
- **Test Loss**: 0.6949

## Examples of Classification
- **Correct Classifications**: High confidence (e.g., 82% and 95%) in correctly identifying same/different individuals.
- **Incorrect Classifications**: Some errors due to visual similarities in facial features or expressions.

## Dependencies
- Python 3.x
- PyTorch
- NumPy

## How to Run
1. Clone the repository.
2. Install required packages:
   ```bash
   pip install torch numpy
   ```
3. Run the training script:
   ```bash
   python main.py
   ```
4. Results and evaluation metrics will be displayed upon completion.
