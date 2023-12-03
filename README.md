# MNIST Image Classification

This project focuses on training a neural network model using the MNIST dataset to classify handwritten digits.

## Usage

### Train model from scratch

To train the model from start to finish:

1. Open the `mnist-classification.ipynb` Jupyter notebook.
2. Run all cells to:
   - Load and preprocess the MNIST dataset.
   - Define the neural network model.
   - Train the model.
   - Evaluate on the test set.
   - Tune hyperparameters.
   - Save the best model to `model.pth`.

### Use pre-trained model for inference

A pre-trained model is provided as `model.pth`. To load it and make predictions:

1. Open a Python shell or script.
2. Import the required libraries:

   ```python
   import torch
   from model import Net
