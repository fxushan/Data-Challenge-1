import argparse
import os
from datetime import datetime
from pathlib import Path
from typing import List

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchsummary import summary
from sklearn.metrics import confusion_matrix, recall_score, precision_score, fbeta_score, roc_auc_score
from sklearn.preprocessing import label_binarize
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import numpy as np
from dc1.net import Net


from dc1.batch_sampler import BatchSampler
from dc1.image_dataset import ImageDataset
from dc1.stochastic_depth import ResNet18, train_with_stochastic_depth, save_model

def load_data():
    train_dataset = ImageDataset(Path("data/X_train.npy"), Path("data/Y_train.npy"))
    test_dataset = ImageDataset(Path("data/X_test.npy"), Path("data/Y_test.npy"))
    return train_dataset, test_dataset

def main(n_epochs):
    # Check if gpu is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # setup code
    train_dataset, test_dataset = load_data()

    # Load the Neural Net. NOTE: set number of distinct labels here
    model = Net(n_classes=6)

    # Initialize optimizer(s) and loss function(s)
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.1)
    loss_function = nn.CrossEntropyLoss()

    # Training without stochastic depth
    model.to(device)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    epochs = 10
    for epoch in range(n_epochs):
        model.train()  # Set the model to train mode
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)

            # Calculate loss
            loss = loss_function(outputs, targets.long())  # Convert targets to torch.long

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Update statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

        # Print statistics at the end of each epoch
        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = 100 * correct / total
        print(f"Epoch [{epoch + 1}/{n_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%")

    print("Training finished.")


    # Training with stochastic depth
    model = ResNet18(prob_skip=0.5)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    model.to(device)

    for epoch in range(epochs):
        losses, accuracies = train_with_stochastic_depth(model, train_loader, optimizer, criterion, device)
        mean_loss = sum(losses) / len(losses)
        mean_accuracy = sum(accuracies) / len(accuracies)
        print(f'Stochastic Depth - Epoch {epoch+1}/{epochs}, Loss: {mean_loss}, Accuracy: {mean_accuracy}')

        # Save accuracy over epochs
        with open('accuracy_stochastic_depth.txt', 'a') as f:
            f.write(f'{mean_accuracy}\n')

        # Save model
        save_model(model, optimizer, epoch, f'model_epoch_{epoch}.pt')

if __name__ == "__main__":
    n_epochs = 10
    main(n_epochs)
