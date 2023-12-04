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
   ```
3. Load the saved model state dict:
   ```python
   model = Net()
   model.load_state_dict(torch.load('model.pth'))
   model.eval()
   ```
4. Preprocess the test image:
   ```python
   transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
   img = transform(Image.open('test_digit.png'))
    ```
5. Pass the input tensor through the model to predict the digit:
   ```python
   with torch.no_grad():
   output = model(img)
   predicted_digit = output.argmax()
   ```

## Files
- mnist-classification.ipynb: Jupyter notebook containing data loading, model training, evaluation, and model saving.
- mnist-classification.html: Replica of jupyter notebook in HTML format for easy access.
- model.pth: Saved state dict of the best-performing model.

## Libraries
- Python
- PyTorch
- Torchvision
- Matplotlib
- Numpy
   
