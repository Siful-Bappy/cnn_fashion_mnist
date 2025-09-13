import torch
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim

import data_reader
import data_loader
import model

torch.manual_seed(42)
os.makedirs("checkpoints", exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

data = data_reader.DataReader()
# print(data.df.shape)

# train test split
x = data.df.iloc[:, 1:].values
y = data.df.iloc[:, 0].values

# training and testing data split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# print("x_train shape:", x_train.shape)
# print("x_test shape:", x_test.shape)

# normalize the data
x_train = x_train / 255.0
x_test = x_test / 255.0

# create custom datasets
train_dataset = data_loader.CustomDataset(x_train, y_train)
test_dataset = data_loader.CustomDataset(x_test, y_test)

# create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, pin_memory=True)

# learning rate & epochs
learning_rate = 0.01
num_epochs = 100

# model, loss, optimizer
# model = model.SimpleCNN(input_feature=1)
model = model.SimpleCNN(input_feature=1).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=1e-4)

# training loop

for epoch in range(num_epochs):
  

  total_epoch_loss = 0

  for batch_features, batch_labels in train_loader:

    # move data to gpu
    batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)

    # forward pass
    outputs = model(batch_features)

    # calculate loss
    loss = criterion(outputs, batch_labels)

    # back pass
    optimizer.zero_grad()
    loss.backward()

    # update grads
    optimizer.step()

    total_epoch_loss = total_epoch_loss + loss.item()

  avg_loss = total_epoch_loss/len(train_loader)
  print(f'Epoch: {epoch + 1} Loss: {avg_loss}')

torch.save(model.state_dict(), 'checkpoints/cnn_fashion_mnist_epoch.pth')
print("Model saved to: checkpoints/cnn_fashion_mnist_epoch.pth")

