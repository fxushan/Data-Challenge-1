# Custom imports
import argparse
import os
from datetime import datetime
from pathlib import Path
from typing import List

# Other imports
import matplotlib.pyplot as plt  # type: ignore
from matplotlib.pyplot import figure
import plotext  # type: ignore
from sklearn.metrics import confusion_matrix, recall_score, precision_score, fbeta_score, roc_auc_score
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split
import seaborn as sns
# Torch imports
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib.pyplot import figure
from torchsummary import summary  # type: ignore
import numpy as np
import sys

from dc_base.batch_sampler_base import BatchSampler
from dc_base.image_dataset_base import ImageDataset
from dc_base.net_base import Net
from dc_base.train_test_base import train_model, test_model, validation_model

from dc_base.evaluation_metrics_base import *


def load_data(X_train_set, Y_train_set):
    train_dataset = ImageDataset(Path(f"data/{X_train_set}"), Path(f"data/{Y_train_set}"))
    test_dataset = ImageDataset(Path("data/X_test.npy"), Path("data/Y_test.npy"))
    X_test = test_dataset.imgs
    y_test = test_dataset.targets

    X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.3, random_state=777, stratify=y_test)
    test_dataset = ImageDataset(X_test, y_test)
    validation_dataset = ImageDataset(X_val, y_val)
    return train_dataset, test_dataset, validation_dataset

class CustomLR:
    def __init__(self, optimizer, decay_rate, stop_epoch):
        self.optimizer = optimizer
        self.decay_rate = decay_rate
        self.stop_epoch = stop_epoch

    def step(self, epoch):
        if epoch < self.stop_epoch:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] *= self.decay_rate

def checkpoint(model, filename):
    torch.save(model.state_dict(), filename)

def resume(model, filename):
    model.load_state_dict(torch.load(filename))

def run_main_base(artifact_path_name, X_train_set, Y_train_set, lr_decay, args: argparse.Namespace, activeloop: bool = True) -> None:
    # Load the train, validation and test data set
    train_dataset, test_dataset, validation_dataset = load_data(X_train_set, Y_train_set)

    # Load the Neural Net. NOTE: set number of distinct labels here
    model = Net(n_classes=6)

    # Initialize optimizer(s) and loss function(s)
    if lr_decay is True:
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
        # scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.gamma)
        scheduler = CustomLR(optimizer, decay_rate=args.gamma, stop_epoch=args.stop_decay)
    else:
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.1)
    loss_function = nn.CrossEntropyLoss()

    # fetch epoch and batch count from arguments
    n_epochs = args.nb_epochs
    batch_size = args.batch_size

    # IMPORTANT! Set this to True to see actual errors regarding
    # the structure of your model (GPU acceleration hides them)!
    # Also make sure you set this to False again for actual model training
    # as training your model with GPU-acceleration (CUDA/MPS) is much faster.
    DEBUG = False

    # Moving our model to the right device (CUDA will speed training up significantly!)
    if torch.cuda.is_available() and not DEBUG:
        print("@@@ CUDA device found, enabling CUDA training...")
        device = "cuda"
        # device = physical_devices[0]
        model.to(device)
        # Creating a summary of our model and its layers:
        summary(model, (1, 128, 128), device=device)
    elif (
            torch.backends.mps.is_available() and not DEBUG
    ):  # PyTorch supports Apple Silicon GPUs from version 1.12
        print("@@@ Apple silicon device enabled, training with Metal backend...")
        device = "mps"
        model.to(device)
    else:
        print("@@@ No GPU boosting device found, training on CPU...")
        device = "cpu"
        # Creating a summary of our model and its layers:
        summary(model, (1, 128, 128), device=device)

    # Let us now train and test our model for multiple epochs:
    train_sampler = BatchSampler(
        batch_size=batch_size, dataset=train_dataset, balanced=args.balanced_batches
    )
    test_sampler = BatchSampler(
        batch_size=100, dataset=test_dataset, balanced=args.balanced_batches
    )

    validation_sampler = BatchSampler(
        batch_size=100, dataset=validation_dataset, balanced=args.balanced_batches
    )

    mean_losses_train: List[torch.Tensor] = []
    mean_losses_validation: List[torch.Tensor] = []
    mean_kappas_train = []
    mean_kappas_validation = []
    mean_mcc_train = []
    mean_mcc_validation = []
    best_loss = -1

    for e in range(n_epochs):
        if activeloop:
            print(f'Epoch {e}...')
            print(f'Learning rate: {optimizer.param_groups[0]["lr"]}')
            # Training:
            losses, kappas, mcc_list, conf_matrix_total_train, cm_total = train_model(model, train_sampler, optimizer,
                                                                            loss_function, device)

            # Calculating and printing statistics:
            # Cross entropy loss
            mean_loss = sum(losses) / len(losses)
            mean_losses_train.append(mean_loss)

            # # Confusion matrix
            # print(conf_matrix_total_train)

            # Cohen's Kappa
            # Average kappa over all batches
            mean_kappa = sum(kappas) / len(kappas)
            mean_kappas_train.append(mean_kappa)

            # Matthew's correlation coefficient
            # Average MCC over all batches
            mean_mcc = sum(mcc_list) / len(mcc_list)
            mean_mcc_train.append(mean_mcc)

            # Validation:
            losses_val, kappas, mcc_list, conf_matrix_total_validation, cm_total = validation_model(model, validation_sampler, loss_function, device)

            # Calculating and printing statistics:
            # Cross entropy loss
            mean_loss_val = sum(losses_val) / len(losses_val)
            mean_losses_validation.append(mean_loss_val)

            # # Confusion matrix
            # print(conf_matrix_total_validation)

            # Cohen's Kappa
            # Average kappa over all batches
            mean_kappa = sum(kappas) / len(kappas)
            mean_kappas_validation.append(mean_kappa)

            # Matthew's correlation coefficient
            # Average MCC over all batches
            mean_mcc = sum(mcc_list) / len(mcc_list)
            mean_mcc_validation.append(mean_mcc)

            # Saving best model & Early stopping
            print(f'Mean loss: {mean_loss_val}')
            if mean_loss_val < best_loss:
                best_loss = mean_loss_val
                best_model_epoch = e
                print(f'New best loss!: {best_loss}')
                torch.save(model.state_dict(), f"model_weights/best_{artifact_path_name}.pth")

            # Learning rate decay step
            if lr_decay is True:
                scheduler.step(e)
            else:
                continue

    # retrieve current time to label artifacts
    now = datetime.now()
    # check if model_weights/ subdir exists
    if not Path("model_weights/").exists():
        os.mkdir(Path("model_weights/"))
    # Check if /artifacts/ subdir exists
    if not Path("artifacts/").exists():
        os.mkdir(Path("artifacts/"))

    # Make directory & path for saving plots on testing data
    if not Path(f"artifacts/{artifact_path_name}/").exists():
        os.mkdir(Path(f"artifacts/{artifact_path_name}/"))
    result_plotting_path: str = f"{artifact_path_name}/{artifact_path_name}"

    # # Saving the model
    # torch.save(model.state_dict(), f"model_weights/model_{now.month:02}_{now.day:02}_{now.hour}_{now.minute:02}.txt")

    # Create and save all plots for training and validation datasets
    plot_cross_entropy_losses(n_epochs, mean_losses_train, mean_losses_validation, result_plotting_path)
    plot_cohens_kappa(n_epochs, mean_kappas_train, mean_kappas_validation, result_plotting_path)
    plot_MCC(n_epochs, mean_mcc_train, mean_mcc_validation, result_plotting_path)
    plot_heatmaps(conf_matrix_total_train, conf_matrix_total_validation, result_plotting_path)

    # Plots for testing dataset:
    losses, kappas, mcc_list, conf_matrix_total_test, cm_total = test_model(model, test_sampler, loss_function, device)
    # Cross Entropy Loss
    mean_loss_test = sum(losses) / len(losses)
    # Cohen's Kappa
    mean_kappa_test = sum(kappas) / len(kappas)
    # Matthew's correlation coefficient
    mean_mcc_test = sum(mcc_list) / len(mcc_list)

    # print(cm_total.stat(summary=True))

    # Save text file with confusion matrix (PyCM library) and all their metrics
    with open(Path("artifacts") / f"{result_plotting_path}_PyCM_test.txt", "a") as f:
        print(f'Mean cross entropy loss: {mean_loss_test}', file=f)
        print(f"Mean Cohen's Kappa: {mean_kappa_test}", file=f)
        print(f"Mean Matthew's correlation coefficient: {mean_mcc_test}", file=f)
        print(f"\n", file=f)

        sys.stdout = f  # Redirect standard output to the file
        print(f'{cm_total}', file=f)
        sys.stdout = sys.__stdout__  # Reset standard output to the console

        print(f'Imbalanced dataset?: {cm_total.imbalance}', file=f)
        print(f'Binary classification?: {cm_total.binary}', file=f)
        print(f'Recommended metrics: {cm_total.recommended_list}', file=f)

        print(f'{cm_total.to_array()}', file=f)

    # Create plot of heatmaps of final confusion matrix
    fig, axs = plt.subplots(figsize=(10, 10))
    labels = ['Etalactasis', 'Effusion', 'Infiltration', 'No Finding', 'Module', 'Pneumothorax']
    sns.heatmap(conf_matrix_total_test, annot=True, fmt='d', xticklabels=labels, yticklabels=labels, ax=axs)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title('Final heatmap for test data')
    plt.savefig(
        Path("artifacts") / f"{result_plotting_path}_ConfusionMatrix_test.png")


    # Early stopping model test
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Net(n_classes=6)
    model.load_state_dict(torch.load(f"model_weights/best_{artifact_path_name}.pth"))
    model.to(device)

    losses_best, kappas_best, mcc_list_best, conf_matrix_total_test_best, cm_total_best = test_model(model,
                                                                                                     test_sampler,
                                                                                                     loss_function,
                                                                                                     device)
    mean_loss_test_best = sum(losses_best) / len(losses_best)
    mean_kappa_test_best = sum(kappas_best) / len(kappas_best)
    mean_mcc_test_best = sum(mcc_list_best) / len(mcc_list_best)

    with open(Path("artifacts") / f"{result_plotting_path}_e{best_model_epoch}_PyCM_test.txt", "a") as f:
        print(f'Mean cross entropy loss: {mean_loss_test_best}', file=f)
        print(f"Mean Cohen's Kappa: {mean_kappa_test_best}", file=f)
        print(f"Mean Matthew's correlation coefficient: {mean_mcc_test_best}", file=f)
        print(f"\n", file=f)

        sys.stdout = f  # Redirect standard output to the file
        print(f'{cm_total_best}', file=f)
        sys.stdout = sys.__stdout__  # Reset standard output to the console

        print(f'Imbalanced dataset?: {cm_total_best.imbalance}', file=f)
        print(f'Binary classification?: {cm_total_best.binary}', file=f)
        print(f'Recommended metrics: {cm_total_best.recommended_list}\n', file=f)

        print(f'{cm_total_best.to_array()}', file=f)

    # Create plot of heatmaps of final confusion matrix
    fig, axs = plt.subplots(figsize=(10, 10))
    labels = ['Etalactasis', 'Effusion', 'Infiltration', 'No Finding', 'Module', 'Pneumothorax']
    sns.heatmap(conf_matrix_total_test_best, annot=True, fmt='d', xticklabels=labels, yticklabels=labels, ax=axs)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title('Final heatmap for test data')
    plt.savefig(
        Path("artifacts") / f"{result_plotting_path}_e{best_model_epoch}_ConfusionMatrix_test.png")

    return conf_matrix_total_test, mean_loss_test, mean_kappa_test, mean_mcc_test

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--nb_epochs", help="number of training iterations", default=10, type=int
    )
    parser.add_argument("--batch_size", help="batch_size", default=25, type=int)
    parser.add_argument(
        "--balanced_batches",
        help="whether to balance batches for class labels",
        default=True,
        type=bool,
    )
    args = parser.parse_args()

    run_main_base(args)


