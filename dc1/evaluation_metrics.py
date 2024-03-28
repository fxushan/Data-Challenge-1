from datetime import datetime
import matplotlib.pyplot as plt  # type: ignore
from matplotlib.pyplot import figure
from pathlib import Path
import seaborn as sns

# retrieve current time to label artifacts
now = datetime.now()

def plot_cross_entropy_losses(n_epochs, mean_losses_train, mean_losses_validation, path):
    # Create plot of Cross Entropy losses
    figure(figsize=(9, 16), dpi=80)
    fig, (ax1, ax2) = plt.subplots(2, sharex=True)
    try:
        ax1.plot(range(1, 1 + n_epochs), [x.detach().cpu() for x in mean_losses_train], label="Train", color="blue")
        ax2.plot(range(1, 1 + n_epochs), [x.detach().cpu() for x in mean_losses_validation], label="Validation", color="red")
    except ValueError:
        ax1.plot(range(1, 2 + n_epochs), [x.detach().cpu() for x in mean_losses_train], label="Train", color="blue")
        ax2.plot(range(1, 2 + n_epochs), [x.detach().cpu() for x in mean_losses_validation], label="Validation", color="red")
    fig.legend()
    fig.suptitle('Cross Entropy loss over epochs')
    fig.savefig(Path("artifacts") / f"{path}_CrossEntropy_train_val.png")

def plot_cohens_kappa(n_epochs, mean_kappas_train, mean_kappas_validation, path):
    # Create plot of Cohen's Kappas
    figure(figsize=(9, 10), dpi=80)
    fig, (ax1, ax2) = plt.subplots(2, sharex=True)
    try:
        ax1.plot(range(1, 1 + n_epochs), [x for x in mean_kappas_train], label="Train", color="blue")
        ax2.plot(range(1, 1 + n_epochs), [x for x in mean_kappas_validation], label="Validation", color="red")
    except ValueError:
        ax1.plot(range(1, 2 + n_epochs), [x for x in mean_kappas_train], label="Train", color="blue")
        ax2.plot(range(1, 2 + n_epochs), [x for x in mean_kappas_validation], label="Validation", color="red")
    fig.legend()
    fig.suptitle("Cohen's Kappa over epochs")
    fig.savefig(Path("artifacts") / f"{path}_CohensKappa_train_val.png")

def plot_MCC(n_epochs, mean_mcc_train, mean_mcc_validation, path):
    # Create plot of Matthews correlation coefficients
    figure(figsize=(9, 10), dpi=80)
    fig, (ax1, ax2) = plt.subplots(2, sharex=True)
    try:
        ax1.plot(range(1, 1 + n_epochs), [x for x in mean_mcc_train], label="Train", color="blue")
        ax2.plot(range(1, 1 + n_epochs), [x for x in mean_mcc_validation], label="Validation", color="red")
    except ValueError:
        ax1.plot(range(1, 2 + n_epochs), [x for x in mean_mcc_train], label="Train", color="blue")
        ax2.plot(range(1, 2 + n_epochs), [x for x in mean_mcc_validation], label="Validation", color="red")
    fig.legend()
    fig.suptitle('Matthews Correlation Coefficient over epochs')
    fig.savefig(Path("artifacts") / f"{path}_MCC_train_val.png")

def plot_heatmaps(conf_matrix_total_train, conf_matrix_total_validation, path):
    fig, axs = plt.subplots(2, figsize=(10, 10))
    sns.heatmap(conf_matrix_total_train, annot=True, fmt='d', ax=axs[0])
    axs[0].set_ylabel('True label')
    axs[0].set_xlabel('Predicted label')
    axs[0].set_title('Final heatmap for train data')
    sns.heatmap(conf_matrix_total_validation, annot=True, fmt='d', ax=axs[1])
    axs[1].set_ylabel('True label')
    axs[1].set_xlabel('Predicted label')
    axs[1].set_title('Final heatmap for validation data')
    fig.savefig(
        Path("artifacts") / f"{path}_ConfusionMatrix_train_val.png")