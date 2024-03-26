# Custom imports
import argparse
import os
from datetime import datetime
from pathlib import Path
from typing import List
import torch.nn.functional as F

# Other imports
import matplotlib.pyplot as plt  # type: ignore
from matplotlib.pyplot import figure
import plotext  # type: ignore
from sklearn.metrics import confusion_matrix, recall_score, precision_score, fbeta_score, roc_auc_score
from sklearn.preprocessing import label_binarize
import seaborn as sns
# Torch imports
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib.pyplot import figure
from torchsummary import summary  # type: ignore
import numpy as np

from dc1.batch_sampler import BatchSampler
from dc1.image_dataset import ImageDataset
from dc1.net import Net
from dc1.train_test import train_model, test_model


def load_data():
    train_dataset = ImageDataset(Path("data/X_train.npy"), Path("data/Y_train.npy"))
    test_dataset = ImageDataset(Path("data/X_test.npy"), Path("data/Y_test.npy"))
    return train_dataset, test_dataset

def test_accuracy(net, device="cpu"):
    testset = ImageDataset(Path("./data/X_test.npy"), Path("./data/Y_test.npy"))
    testloader = torch.utils.data.DataLoader(testset, batch_size=16, shuffle=False, num_workers=2)
    correct = 0
    total = 0
    true_labels = []
    predicted_labels = []
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            probabilities = F.softmax(outputs, dim=1)  # Apply softmax to convert outputs to probabilities
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            true_labels.extend(labels.cpu().numpy())
            predicted_labels.extend(predicted.cpu().numpy())
            print(probabilities.cpu().numpy())  # Print the softmax probabilities
    return (correct / total, true_labels, predicted_labels)

def perf_measure(y_actual, y_hat):
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    for i in range(len(y_hat)):
        if y_actual[i] == y_hat[i] == 3:
            TN += 1
        elif y_actual[i] == y_hat[i]:
            TP += 1
        if y_hat[i] == 3 and y_actual[i] != y_hat[i]:
            FN += 1
        elif y_actual[i] != y_hat[i]:
            FP += 1
    return (TP, FP, TN, FN)

# def heatmap_plot(y, predictions):
#     labels = ['Etalactasis', 'Effusion', 'Infiltration', 'No Finding', 'Module', 'Pneumothorax']
#     cm = confusion_matrix(y, predictions)
#     sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels)
#     plt.xlabel('Predicted')
#     plt.ylabel('True')
#     plt.title('Confusion Matrix')
#     plt.show()
def calculate_additional_metrics(y_true, y_pred):
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f2_score = fbeta_score(y_true, y_pred, beta=2, average='weighted')

    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F2 Score: {f2_score:.4f}")

def main(args: argparse.Namespace, activeloop: bool = True) -> None:
    # Load the train and test data set
    #train_dataset = ImageDataset(Path("data/X_train.npy"), Path("data/Y_train.npy"))
    #test_dataset = ImageDataset(Path("data/X_test.npy"), Path("data/Y_test.npy"))
    train_dataset, test_dataset = load_data()

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
            losses, kappas, mcc_list, conf_matrix_total_train, cm_total = train_model(model, train_sampler, optimizer,
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
            print(f"\nEpoch {e + 1} training done, average Cohen's kappa train set: {mean_kappa}")

            # Matthew's correlation coefficient
            # Average MCC over all batches
            mean_mcc = sum(mcc_list) / len(mcc_list)
            mean_mcc_train.append(mean_mcc)
            print(f"Epoch {e + 1} training done, average MCC train set: {mean_mcc}\n")

            # Testing:
            losses, kappas, mcc_list, conf_matrix_total_test, cm_total = test_model(model, test_sampler, loss_function, device)

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
            print(f"\nEpoch {e + 1} testing done, average Cohen's kappa test set: {mean_kappa}")

            # Matthew's correlation coefficient
            # Average MCC over all batches
            mean_mcc = sum(mcc_list) / len(mcc_list)
            mean_mcc_test.append(mean_mcc)
            print(f"Epoch {e + 1} testing done, average MCC test set: {mean_mcc}\n")

            # # Plotting during training
            # plotext.clf()
            # plotext.scatter(mean_losses_train, label="train")
            # plotext.scatter(mean_losses_test, label="test")
            # plotext.title("Train and test loss")
            #
            # plotext.xticks([i for i in range(len(mean_losses_train) + 1)])
            #
            # plotext.show()

    accuracy, true_labels, predicted_labels = test_accuracy(model, device)
    print(f'Test Accuracy: {accuracy * 100:.2f}%')

    # Calculate AUC-ROC score
    y_one_hot = label_binarize(true_labels, classes=[0, 1, 2, 3, 4, 5])
    y_pred_one_hot = label_binarize(predicted_labels, classes=[0, 1, 2, 3, 4, 5])
    auc_roc = roc_auc_score(y_one_hot, y_pred_one_hot, average="macro")
    print(f'AUC-ROC Score: {auc_roc}')

    TP, FP, TN, FN = perf_measure(true_labels, predicted_labels)
    print(f'TP={TP} FP={FP} TN={TN} FN={FN}')

    # heatmap_plot(true_labels, predicted_labels)

    # Calculate accuracy
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_sampler:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print(f"Accuracy: {100 * accuracy:.2f}%")

    # retrieve current time to label artifacts
    now = datetime.now()
    # check if model_weights/ subdir exists
    if not Path("model_weights/").exists():
        os.mkdir(Path("model_weights/"))

    # Saving the model
    torch.save(model.state_dict(), f"model_weights/model_{now.month:02}_{now.day:02}_{now.hour}_{now.minute:02}.txt")

    # Create plot of Cross Entropy losses
    figure(figsize=(9, 16), dpi=80)
    fig, (ax1, ax2) = plt.subplots(2, sharex=True)

    ax1.plot(range(1, 1 + n_epochs), [x.detach().cpu() for x in mean_losses_train], label="Train", color="blue")
    ax2.plot(range(1, 1 + n_epochs), [x.detach().cpu() for x in mean_losses_test], label="Test", color="red")
    fig.legend()
    fig.suptitle('Cross Entropy loss over epochs')

    #Accuracy

    accuracy, true_labels, predicted_labels = test_accuracy(model, device)
    print(f'Test Accuracy: {accuracy*100:.2f}%')

    #Precision, Recall and F2 score

    calculate_additional_metrics(true_labels, predicted_labels)
    # heatmap_plot(true_labels, predicted_labels)
    
    # Check if /artifacts/ subdir exists
    if not Path("artifacts/").exists():
        os.mkdir(Path("artifacts/"))

    # save plot of losses
    fig.savefig(Path("artifacts") / f"session_{now.month:02}_{now.day:02}_{now.hour}_{now.minute:02}_CrossEntropy.png")

    # Create plot of Cohen's Kappas
    figure(figsize=(9, 10), dpi=80)
    fig, (ax1, ax2) = plt.subplots(2, sharex=True)

    ax1.plot(range(1, 1 + n_epochs), [x for x in mean_kappas_train], label="Train", color="blue")
    ax2.plot(range(1, 1 + n_epochs), [x for x in mean_kappas_test], label="Test", color="red")
    fig.legend()
    fig.suptitle("Cohen's Kappa over epochs")

    fig.savefig(Path("artifacts") / f"session_{now.month:02}_{now.day:02}_{now.hour}_{now.minute:02}_CohensKappa.png")

    # Create plot of Matthews correlation coefficients
    figure(figsize=(9, 10), dpi=80)
    fig, (ax1, ax2) = plt.subplots(2, sharex=True)

    ax1.plot(range(1, 1 + n_epochs), [x for x in mean_mcc_train], label="Train", color="blue")
    ax2.plot(range(1, 1 + n_epochs), [x for x in mean_mcc_test], label="Test", color="red")
    fig.legend()
    fig.suptitle('Matthews Correlation Coefficient over epochs')

    fig.savefig(Path("artifacts") / f"session_{now.month:02}_{now.day:02}_{now.hour}_{now.minute:02}_MCC.png")

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
        Path("artifacts") / f"session_{now.month:02}_{now.day:02}_{now.hour}_{now.minute:02}_ConfusionMatrix.png")

    # Save text file with confusion matrix (PyCM library) and all their metrics
    with open(Path("artifacts") / f"session_{now.month:02}_{now.day:02}_{now.hour}_{now.minute:02}_PyCM.txt", "a") as f:
        # Printing CM and metrics gives None, so I manually copy it over
        print(f'{cm_total.print_matrix()}', file=f)
        print(f'{cm_total.stat(summary=True)}', file=f)

        print(f'Imbalanced dataset?: {cm_total.imbalance}', file=f)
        print(f'Binary classification?: {cm_total.binary}', file=f)
        print(f'Recommended metrics: {cm_total.recommended_list}', file=f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--nb_epochs", help="number of training iterations", default=34, type=int
    )
    parser.add_argument("--batch_size", help="batch_size", default=10, type=int)
    parser.add_argument(
        "--balanced_batches",
        help="whether to balance batches for class labels",
        default=True,
        type=bool,
    )
    args = parser.parse_args()

    main(args)
