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
   from PIL import Image
   ```
3. Load the saved model state dict:
   ```python
   model = Net()
   model.load_state_dict(torch.load('model.pth'))
   model.eval()
   ```
4. Preprocess the test image:
   ```python
   test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]) #transforms.Resize((784, 512)),
   img = Image.open('91.jpg')
   img = img.convert('L')
   width, height = img.size
   aspect_ratio = width / float(height)
   resized_img = img.resize((28, 28), Image.Resampling.LANCZOS )
   img_fi = test_transform(resized_img)
    ```
5. Visulaze the image
   ```python
   img_np = torch.stack((imageOne,)).numpy()
   img_np = img_np.squeeze()
   plt.imshow(img_np, cmap='gray')
   plt.axis('off')
   plt.show()
   ```
6. Pass the input tensor through the model to predict the digit:
   ```python
   with torch.no_grad():
    output = inf(img_fi)
    predicted_digit = output.argmax()
    print("Predicted Digit: ", predicted_digit.item())
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
   
