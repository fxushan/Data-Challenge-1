# Custom imports
import argparse
import os
import sys
from typing import List

# Other imports
import matplotlib.pyplot as plt  # type: ignore
import numpy as np
import plotext  # type: ignore
# Torch imports
import torch
import torch.nn as nn
import torch.optim as optim
from pycm import ConfusionMatrix, Compare
from sklearn.model_selection import train_test_split
from torchsummary import summary  # type: ignore

from dc1.batch_sampler import BatchSampler
from dc1.image_dataset import ImageDataset
from dc1.net import Net
from dc1.train_test import train_model, test_model, validation_model
from dc_base.main_base import run_main_base
from evaluation_metrics import *


def load_data(X_train_set, Y_train_set):
    train_dataset = ImageDataset(Path(f"data/{X_train_set}"), Path(f"data/{Y_train_set}"))
    test_dataset = ImageDataset(Path("data/X_test.npy"), Path("data/Y_test.npy"))
    X_test = test_dataset.imgs
    y_test = test_dataset.targets

    X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.3, random_state=777, stratify=y_test)
    test_dataset = ImageDataset(X_test, y_test)
    validation_dataset = ImageDataset(X_val, y_val)
    return train_dataset, test_dataset, validation_dataset


def main(artifact_path_name, X_train_set, Y_train_set, args: argparse.Namespace, activeloop: bool = True) -> None:
    # Load the train, validation and test data set
    train_dataset, test_dataset, validation_dataset = load_data(X_train_set, Y_train_set)

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

    validation_sampler = BatchSampler(
        batch_size=100, dataset=validation_dataset, balanced=args.balanced_batches
    )

    mean_losses_train: List[torch.Tensor] = []
    mean_losses_validation: List[torch.Tensor] = []
    mean_kappas_train = []
    mean_kappas_validation = []
    mean_mcc_train = []
    mean_mcc_validation = []

    for e in range(n_epochs):
        if activeloop:
            print(f'Epoch {e}...')
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
            losses, kappas, mcc_list, conf_matrix_total_validation, cm_total = validation_model(model,
                                                                                                validation_sampler,
                                                                                                loss_function, device)

            # Calculating and printing statistics:
            # Cross entropy loss
            mean_loss = sum(losses) / len(losses)
            mean_losses_validation.append(mean_loss)

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

    # Create plot of heatmaps of final confusion matrix
    fig, axs = plt.subplots(figsize=(10, 10))
    labels = ['Etalactasis', 'Effusion', 'Infiltration', 'No Finding', 'Module', 'Pneumothorax']
    sns.heatmap(conf_matrix_total_test, annot=True, fmt='d', xticklabels=labels, yticklabels=labels, ax=axs)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title('Final heatmap for test data')
    plt.savefig(
        Path("artifacts") / f"{result_plotting_path}_ConfusionMatrix_test.png")

    return conf_matrix_total_test, mean_loss_test, mean_kappa_test, mean_mcc_test


def run_main_x_times(x, artifact_path_name, X_train_set, Y_train_set, base, args: argparse.Namespace):
    losses = []
    kappas = []
    mcc = []
    for i in range(x):
        print(f'Run: {i}\n')
        if i == 0:
            if base is True:
                cm_sum, mean_loss_test, mean_kappa_test, mean_mcc_test  = (
                    run_main_base(f'{artifact_path_name}_{i}', X_train_set, Y_train_set, args))
            else:
                cm_sum, mean_loss_test, mean_kappa_test, mean_mcc_test = (
                    main(f'{artifact_path_name}_{i}', X_train_set, Y_train_set, args))
            losses.append(mean_loss_test)
            kappas.append(mean_kappa_test)
            mcc.append(mean_mcc_test)
        elif i != 0:
            if base is True:
                cm_total, mean_loss_test, mean_kappa_test, mean_mcc_test = (
                    run_main_base(f'{artifact_path_name}_{i}', X_train_set, Y_train_set, args))
            else:
                cm_total, mean_loss_test, mean_kappa_test, mean_mcc_test = (
                    main(f'{artifact_path_name}_{i}', X_train_set, Y_train_set, args))
            cm_sum = np.add(cm_sum, cm_total)
            losses.append(mean_loss_test)
            kappas.append(mean_kappa_test)
            mcc.append(mean_mcc_test)

    mean_loss = sum(losses) / len(losses)
    mean_kappa = sum(kappas) / len(kappas)
    mean_mcc = sum(mcc) / len(mcc)

    cm_avg = cm_sum / x
    cm_avg = cm_avg.astype(int)

    cm_dict = {str(i): {str(j): cm_avg[i, j] for j in range(cm_avg.shape[1])} for i in
               range(cm_avg.shape[0])}
    pycm_avg = ConfusionMatrix(matrix=cm_dict)

    if not Path(f"artifacts/{artifact_path_name}/").exists():
        os.mkdir(Path(f"artifacts/{artifact_path_name}/"))

    # Save text file with confusion matrix (PyCM library) and all their metrics
    with open(Path("artifacts") / f"{artifact_path_name}/{artifact_path_name}_PyCM_test_avg.txt", "a") as f:
        print(f'Results {artifact_path_name}, as an average from {x} different trainings', file=f)
        print(f'Average cross entropy loss: {mean_loss}', file=f)
        print(f"Average Cohen's Kappa: {mean_kappa}", file=f)
        print(f"Average Matthew's correlation coefficient: {mean_mcc}", file=f)
        print(f"\n", file=f)

        sys.stdout = f  # Redirect standard output to the file
        print(f'{pycm_avg}', file=f)
        sys.stdout = sys.__stdout__  # Reset standard output to the console

        print(f'Imbalanced dataset?: {pycm_avg.imbalance}', file=f)
        print(f'Binary classification?: {pycm_avg.binary}', file=f)
        print(f'Recommended metrics: {pycm_avg.recommended_list}', file=f)

    # Create plot of heatmaps of final confusion matrix
    fig, axs = plt.subplots(figsize=(10, 10))
    labels = ['Etalactasis', 'Effusion', 'Infiltration', 'No Finding', 'Module', 'Pneumothorax']
    sns.heatmap(cm_avg, annot=True, fmt='d', xticklabels=labels, yticklabels=labels, ax=axs)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title('Final heatmap for test data')
    plt.savefig(
        Path("artifacts") / f"{artifact_path_name}/{artifact_path_name}_CM_test_avg.png")

    return cm_dict


if __name__ == "__main__":
    # Base run
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
    artifact_path_name_1 = "New_E10_BS25"
    pycm_avg_1 = (
        run_main_x_times(20, artifact_path_name_1, "X_train.npy", "Y_train.npy", False, args))

    # Test run
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
    artifact_path_name_2 = "New_E34_BS10"
    pycm_avg_2 = (
        run_main_x_times(20, artifact_path_name_2, "X_train.npy", "Y_train.npy", False, args))


    total_1 = sum(sum(inner_dict.values()) for inner_dict in pycm_avg_1.values())
    total_2 = sum(sum(inner_dict.values()) for inner_dict in pycm_avg_2.values())
    difference = total_1 - total_2
    pycm_avg_2['0']['0'] += difference

    pycm_avg_1 = ConfusionMatrix(matrix=pycm_avg_1)
    pycm_avg_2 = ConfusionMatrix(matrix=pycm_avg_2)
    compare = Compare({f'{artifact_path_name_1}': pycm_avg_1, f'{artifact_path_name_2}': pycm_avg_2})

    with open(Path("artifacts") / f"Comparisons/{artifact_path_name_1} vs {artifact_path_name_2}.txt", "a") as f:
        print(f'Comparison of PyCM confusion matrices of the new model with 10 epochs and 25 batch size vs. new model with 34 epochs and 10 batch size', file=f)
        print(f'Both models are trained 20 times, with the averages as final results.', file=f)
        print(f'', file=f)

        sys.stdout = f  # Redirect standard output to the file
        print(f'{compare}', file=f)
        sys.stdout = sys.__stdout__  # Reset standard output to the console




    # # Base run
    # parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     "--nb_epochs", help="number of training iterations", default=10, type=int
    # )
    # parser.add_argument("--batch_size", help="batch_size", default=25, type=int)
    # parser.add_argument(
    #     "--balanced_batches",
    #     help="whether to balance batches for class labels",
    #     default=True,
    #     type=bool,
    # )
    # args = parser.parse_args()
    # artifact_path_name_1 = "Base_No_Aug"
    # pycm_avg_1 = (
    #     run_main_x_times(20, artifact_path_name_1, "X_train.npy", "Y_train.npy", True, args))
    #
    # # Test run
    # artifact_path_name_2 = "Base_tensorflow_aug"
    # pycm_avg_2 = (
    #     run_main_x_times(20, artifact_path_name_2, "X_tensor.npy", "Y_tensor.npy", True, args))
    #
    #
    # total_1 = sum(sum(inner_dict.values()) for inner_dict in pycm_avg_1.values())
    # total_2 = sum(sum(inner_dict.values()) for inner_dict in pycm_avg_2.values())
    # difference = total_1 - total_2
    # pycm_avg_2['0']['0'] += difference
    #
    # pycm_avg_1 = ConfusionMatrix(matrix=pycm_avg_1)
    # pycm_avg_2 = ConfusionMatrix(matrix=pycm_avg_2)
    # compare = Compare({f'{artifact_path_name_1}': pycm_avg_1, f'{artifact_path_name_2}': pycm_avg_2})
    #
    # with open(Path("artifacts") / f"Comparisons/{artifact_path_name_1} vs {artifact_path_name_2}.txt", "a") as f:
    #     print(f'Comparison of PyCM confusion matrices of the base model (no augmentation) vs base model (tensorflow augmentation)', file=f)
    #     print(f'Epochs: 10. Batch Size: 25', file=f)
    #     print(f'Both models are trained 20 times, with the averages as final results.', file=f)
    #     print(f'', file=f)
    #
    #     sys.stdout = f  # Redirect standard output to the file
    #     print(f'{compare}', file=f)
    #     sys.stdout = sys.__stdout__  # Reset standard output to the console