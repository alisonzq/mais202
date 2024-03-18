import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
import torchvision.transforms as transforms
import torch.optim as optim
from torchvision.datasets import ImageFolder 
from torch.utils.data import DataLoader, random_split

import torchmetrics

torch.manual_seed(0)
np.random.seed(0)

# Create 1 random image of shape (1, 28, 28)
image = torch.rand(1, 28, 28)

# Create a convolutional that applies 6 3x3 filters tso get 6 output channels (Stride = 1, same padding)
conv_filters = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=(3, 3), stride=1, padding="same")

# Convolve the image with the filters
output = conv_filters(image)
print(output.shape)

# We can create an image
image = torch.rand(1, 6, 6)

# Apply a 2x2 max pool filter
output = F.max_pool2d(image, 2)
print(output.shape)


# Define the path to your dataset
data_dir = 'garbage_classification'

# Define your transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to (224, 224)
    transforms.ToTensor(),           # Convert images to PyTorch tensors
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize using ImageNet mean and std
])

# Load the dataset
dataset = ImageFolder(root=data_dir, transform=transform)

# Define the size of the training and validation sets
validation_split = 0.2
test_split = 0.1
test_size = int(len(dataset) * test_split)
val_size = int((len(dataset) - test_size) * validation_split)
train_size = len(dataset) - val_size - test_size

# Split the dataset into training, validation, and test sets
train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])

# Define your data loaders
batch_size = 128
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

# Print dataset sizes
print(f"Training set size: {len(train_set)}")
print(f"Validation set size: {len(val_set)}")
print(f"Test set size: {len(test_set)}")

# If a GPU is available, use it
if torch.cuda.is_available():
  DEVICE = torch.device("cuda")

# Else, revert to the default (CPU)
else:
  DEVICE = torch.device("cpu")

print(DEVICE)

loss = nn.CrossEntropyLoss().to(DEVICE) # Since we are doing multiclass classification
accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=8).to(DEVICE) # Regular accuracy

class CNN(nn.Module):
  '''
    Class representing a CNN with 2 (convolutional + activation + maxpooling) layers, connected to a single linear layer for prediction
  '''
  def __init__(self):
    super(CNN, self).__init__()
    self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding="same")
    self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding="same")
    self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding="same")
    self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding="same")
    self.conv5 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding="same")
       
    self.linear = nn.Linear(100352, 8) 

  def forward(self, x):
    '''Forward pass function, needs to be defined for every model'''

    x = self.conv1(x)
    x = F.relu(x)
    x = F.max_pool2d(x, 2) # 2x2 maxpool

    x = self.conv2(x)
    x = F.relu(x)
    x = F.max_pool2d(x, 2)

    x = self.conv3(x)
    x = F.relu(x)
    x = F.max_pool2d(x, 2)

    x = torch.flatten(x, start_dim = 1) # Flatten to a 1D vector
    x = self.linear(x)
    x = F.softmax(x, dim = 1) # dim = 1 to softmax along the rows of the output (We want the probabilities of all classes to sum up to 1)

    return x
  
lr = 3e-5
num_epochs = 20 
num_iterations_before_validation = 1000 # We will compute the validation accuracy every 1000 iterations

cnn_metrics = {}
cnn_models = {}


cnn_metrics[lr] = {
    "accuracies": [],
    "losses": []
}

cnn = CNN().to(DEVICE)
optimizer = optim.Adam(cnn.parameters(), lr)
cnn_models[lr] = cnn

# Iterate through the epochs
for epoch in range(num_epochs):

    # Iterate through the training data
    for iteration, (X_train, y_train) in enumerate(train_loader):

        # Move the batch to GPU if it's available
        X_train = X_train.to(DEVICE)
        y_train = y_train.to(DEVICE)

        # The optimizer accumulates the gradient of each weight as we do forward passes -> zero_grad resets all gradients to 0
        optimizer.zero_grad()

        # Compute a forward pass and make a prediction
        y_hat = cnn(X_train)

        # Compute the loss
        train_loss = loss(y_hat, y_train)

        # Compute the gradients in the optimizer
        train_loss.backward()

        # Update the weights
        optimizer.step()

        # Check if should compute the validation metrics for plotting later
        if iteration % num_iterations_before_validation == 0:

        # Don't compute gradients on the validation set
            with torch.no_grad():

                # Keep track of the losses & accuracies
                val_accuracy_sum = 0
                val_loss_sum = 0

                # Make a predictions on the full validation set, batch by batch
                for X_val, y_val in val_loader:
                    # Move the batch to GPU if it's available
                    X_val = X_val.to(DEVICE)
                    y_val = y_val.to(DEVICE)

                    y_hat = cnn(X_val)
                    val_accuracy_sum += accuracy(y_hat, y_val)
                    val_loss_sum += loss(y_hat, y_val)

                # Divide by the number of iterations (and move back to CPU)
                val_accuracy = (val_accuracy_sum / len(val_loader)).cpu()
                val_loss = (val_loss_sum / len(val_loader)).cpu()

                # Store the values in the dictionary
                cnn_metrics[lr]["accuracies"].append(val_accuracy)
                cnn_metrics[lr]["losses"].append(val_loss)

                # Print to console
                print(f"LR = {lr} --- EPOCH = {epoch} --- ITERATION = {iteration}")
                print(f"Validation loss = {val_loss} --- Validation accuracy = {val_accuracy}")


test_accuracy_sum = 0
best_cnn = cnn_models[lr] # Extract the best trained model

for X_test, y_test in test_loader:

  # Move batch to GPU
  X_test = X_test.to(DEVICE)
  y_test = y_test.to(DEVICE)

  # Make prediction
  y_hat = best_cnn(X_test)

  # Calculate accuracy
  test_accuracy_sum += accuracy(y_hat, y_test)

# Divide by the number of batches in the test set
test_accuracy = (test_accuracy_sum / len(test_loader)).cpu()
print(f"The test set accuracy of the best model is: {test_accuracy}")


total_params_cnn = sum(
	param.numel() for param in best_cnn.parameters()
)

print(f"Number of weights in the CNN: {total_params_cnn}")

