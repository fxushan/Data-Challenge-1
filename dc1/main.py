# Custom imports
import argparse
import os
from datetime import datetime
from pathlib import Path
from typing import List

# Other imports
import matplotlib.pyplot as plt  # type: ignore
import plotext  # type: ignore
import seaborn as sns
# Torch imports
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib.pyplot import figure
from torchsummary import summary  # type: ignore

from dc1.batch_sampler import BatchSampler
from dc1.image_dataset import ImageDataset
from dc1.net import Net
from dc1.train_test import train_model, test_model


def main(args: argparse.Namespace, activeloop: bool = True) -> None:
    # Load the train and test data set
    train_dataset = ImageDataset(Path("data/X_train.npy"), Path("data/Y_train.npy"))
    test_dataset = ImageDataset(Path("data/X_test.npy"), Path("data/Y_test.npy"))

    # Load the Neural Net. NOTE: set number of distinct labels here
    model = Net(n_classes=6)

    # Initialize optimizer(s) and loss function(s)
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

    mean_losses_train: List[torch.Tensor] = []
    mean_losses_test: List[torch.Tensor] = []
    mean_kappas_train = []
    mean_kappas_test = []
    mean_mcc_train = []
    mean_mcc_test = []

    for e in range(n_epochs):
        if activeloop:
            # Training:
            losses, kappas, mcc_list, conf_matrix_total_train = train_model(model, train_sampler, optimizer,
                                                                            loss_function, device)

            # Calculating and printing statistics:
            # Cross entropy loss
            mean_loss = sum(losses) / len(losses)
            mean_losses_train.append(mean_loss)
            print(f"\nEpoch {e + 1} training done, loss on train set: {mean_loss}\n")

            # Confusion matrix
            print(conf_matrix_total_train)

            # Cohen's Kappa
            # Average kappa over all batches
            mean_kappa = sum(kappas) / len(kappas)
            mean_kappas_train.append(mean_kappa)
            print(f"\nEpoch {e + 1} training done, average Cohen's kappa train set: {mean_kappa}\n")

            # Matthew's correlation coefficient
            # Average MCC over all batches
            mean_mcc = sum(mcc_list) / len(mcc_list)
            mean_mcc_train.append(mean_mcc)
            print(f"\nEpoch {e + 1} training done, average MCC train set: {mean_mcc}\n")

            #     # Cohen's kappa over final confusion matrix
            # Po = sum(conf_matrix_total[i][i] for i in range(6)) / np.sum(conf_matrix_total)
            # row_sums = [sum(row) for row in conf_matrix_total]
            # col_sums = [sum(col) for col in zip(*conf_matrix_total)]
            # Pe = sum((row_sums[i] * col_sums[i]) for i in range(6)) / (np.sum(conf_matrix_total) ** 2)
            # kappa_conf_matrix = (Po - Pe) / (1 - Pe)
            # print(f"\nEpoch {e + 1} training done, kappa over confusion matrix in train set: {kappa_conf_matrix}\n")

            # Testing:
            losses, kappas, mcc_list, conf_matrix_total_test = test_model(model, test_sampler, loss_function, device)

            # Calculating and printing statistics:
            # Cross entropy loss
            mean_loss = sum(losses) / len(losses)
            mean_losses_test.append(mean_loss)
            print(f"\nEpoch {e + 1} testing done, loss on test set: {mean_loss}\n")

            # Confusion matrix
            print(conf_matrix_total_test)

            # Cohen's Kappa
            # Average kappa over all batches
            mean_kappa = sum(kappas) / len(kappas)
            mean_kappas_test.append(mean_kappa)
            print(f"\nEpoch {e + 1} training done, average Cohen's kappa test set: {mean_kappa}\n")

            # Matthew's correlation coefficient
            # Average MCC over all batches
            mean_mcc = sum(mcc_list) / len(mcc_list)
            mean_mcc_test.append(mean_mcc)
            print(f"\nEpoch {e + 1} training done, average MCC test set: {mean_mcc}\n")

            # Plotting during training
            plotext.clf()
            plotext.scatter(mean_losses_train, label="train")
            plotext.scatter(mean_losses_test, label="test")
            plotext.title("Train and test loss")

            plotext.xticks([i for i in range(len(mean_losses_train) + 1)])

            plotext.show()

    # retrieve current time to label artifacts
    now = datetime.now()
    # check if model_weights/ subdir exists
    if not Path("model_weights/").exists():
        os.mkdir(Path("model_weights/"))

    # Saving the model
    torch.save(model.state_dict(), f"model_weights/model_{now.month:02}_{now.day:02}_{now.hour}_{now.minute:02}.txt")

    # Create plot of losses
    figure(figsize=(9, 16), dpi=80)
    fig, (ax1, ax2) = plt.subplots(2, sharex=True)

    ax1.plot(range(1, 1 + n_epochs), [x.detach().cpu() for x in mean_losses_train], label="Train", color="blue")
    ax2.plot(range(1, 1 + n_epochs), [x.detach().cpu() for x in mean_losses_test], label="Test", color="red")
    fig.legend()
    fig.suptitle('Cross Entropy loss over epochs')

    # Check if /artifacts/ subdir exists
    if not Path("artifacts/").exists():
        os.mkdir(Path("artifacts/"))

    # save plot of losses
    fig.savefig(Path("artifacts") / f"CrossEntropy_session_{now.month:02}_{now.day:02}_{now.hour}_{now.minute:02}.png")

    # Create plot of heatmaps of final confusion matrix
    fig, axs = plt.subplots(2, figsize=(10, 10))

    sns.heatmap(conf_matrix_total_train, annot=True, fmt='d', ax=axs[0])
    axs[0].set_ylabel('True label')
    axs[0].set_xlabel('Predicted label')
    axs[0].set_title('Final heatmap for train data')

    sns.heatmap(conf_matrix_total_test, annot=True, fmt='d', ax=axs[1])
    axs[1].set_ylabel('True label')
    axs[1].set_xlabel('Predicted label')
    axs[1].set_title('Final heatmap for test data')

    fig.savefig(
        Path("artifacts") / f"ConfusionMatrix_session_{now.month:02}_{now.day:02}_{now.hour}_{now.minute:02}.png")

    # Create plot of Cohen's Kappas
    figure(figsize=(9, 10), dpi=80)
    fig, (ax1, ax2) = plt.subplots(2, sharex=True)

    ax1.plot(range(1, 1 + n_epochs), [x for x in mean_kappas_train], label="Train", color="blue")
    ax2.plot(range(1, 1 + n_epochs), [x for x in mean_kappas_test], label="Test", color="red")
    fig.legend()
    fig.suptitle("Cohen's Kappa over epochs")

    fig.savefig(Path("artifacts") / f"CohenKappa_session_{now.month:02}_{now.day:02}_{now.hour}_{now.minute:02}.png")

    # Create plot of Matthews correlation coefficients
    figure(figsize=(9, 10), dpi=80)
    fig, (ax1, ax2) = plt.subplots(2, sharex=True)

    ax1.plot(range(1, 1 + n_epochs), [x for x in mean_mcc_train], label="Train", color="blue")
    ax2.plot(range(1, 1 + n_epochs), [x for x in mean_mcc_test], label="Test", color="red")
    fig.legend()
    fig.suptitle('Matthews Correlation Coefficient over epochs')

    fig.savefig(Path("artifacts") / f"MCC_session_{now.month:02}_{now.day:02}_{now.hour}_{now.minute:02}.png")


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

    main(args)
