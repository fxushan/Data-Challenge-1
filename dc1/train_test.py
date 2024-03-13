from typing import Callable, List, Any

import torch
from sklearn.metrics import cohen_kappa_score, confusion_matrix, matthews_corrcoef
from pycm import ConfusionMatrix
from tensorflow import Tensor
from tqdm import tqdm

from dc1.batch_sampler import BatchSampler
from dc1.net import Net


def train_model(
        model: Net,
        train_sampler: BatchSampler,
        optimizer: torch.optim.Optimizer,
        loss_function: Callable[..., torch.Tensor],
        device: str,
) -> tuple[List[Tensor], List[float], List[float], int | Any]:
    #
    # Let us keep track of all the losses:
    losses = []
    kappas = []
    mcc_list = []
    conf_matrix_total = 0
    cm_total = 0
    batch_number = 0
    # Put the model in train mode:
    model.train()
    # Feed all the batches one by one:
    for batch in tqdm(train_sampler):
        # Get a batch:
        x, y = batch
        # Making sure our samples are stored on the same device as our model:
        x, y = x.to(device), y.to(device)
        # Get predictions:
        predictions = model.forward(x)

        # Get probabilities and predicted classes
        probabilities = torch.nn.functional.softmax(predictions, dim=1)
        _, predicted_classes = torch.max(probabilities, dim=1)

        # Accuracy metric
        accuracy = (predicted_classes == y).float().mean()

        # Cross entropy
        loss = loss_function(predictions, y)
        losses.append(loss)

        # Cohen's kappa
        predictions_np = predicted_classes.detach().cpu().numpy()
        labels_np = y.detach().cpu().numpy()
        kappa = cohen_kappa_score(predictions_np, labels_np)
        kappas.append(kappa)

        # Matthew's Correlation Coefficient
        mcc = matthews_corrcoef(predictions_np, labels_np)
        mcc_list.append(mcc)

        # print(predictions)
        # print(probabilities)
        # print('\n')
        # print(predicted_classes)
        # print(y)
        #
        # print(f'Accuracy: {accuracy}')
        # print(f'Cross entropy: {loss}')
        # print(f"Cohen's Kappa: {kappa}")
        # print(f"MCC: {mcc}")

        # Confusion Matrix
        labels = list(range(0, 6))
        conf_matrix_batch = confusion_matrix(y_pred=predictions_np, y_true=labels_np, labels=labels)
        # print(conf_matrix_batch)
        conf_matrix_total = conf_matrix_batch + conf_matrix_total

        # Confusion Matrix 2
        if batch_number == 0:
            cm_total = ConfusionMatrix(actual_vector=labels_np, predict_vector=predictions_np, classes=[0, 1, 2, 3, 4, 5])
            batch_number += 1
        elif batch_number != 0:
            cm = ConfusionMatrix(actual_vector=labels_np, predict_vector=predictions_np, classes=[0, 1, 2, 3, 4, 5])
            cm_total = cm_total.combine(cm)

        # We first need to make sure we reset our optimizer at the start.
        # We want to learn from each batch separately,
        # not from the entire dataset at once.
        optimizer.zero_grad()
        # We now back-propagate our loss through our model:
        loss.backward()
        # We then make the optimizer take a step in the right direction.
        optimizer.step()
    return losses, kappas, mcc_list, conf_matrix_total, cm_total


def test_model(
        model: Net,
        test_sampler: BatchSampler,
        loss_function: Callable[..., torch.Tensor],
        device: str,
) -> tuple[List[Tensor], List[float], List[float], int | Any]:
    # Setting the model to evaluation mode:
    model.eval()
    losses = []
    kappas = []
    mcc_list = []
    conf_matrix_total = 0
    cm_total = 0
    batch_number = 0
    # We need to make sure we do not update our model based on the test data:
    with torch.no_grad():
        for (x, y) in tqdm(test_sampler):
            # Making sure our samples are stored on the same device as our model:
            x = x.to(device)
            y = y.to(device)
            prediction = model.forward(x)

            # Get probabilities and predicted classes
            probabilities = torch.nn.functional.softmax(prediction, dim=1)
            _, predicted_classes = torch.max(probabilities, dim=1)

            # Cross entropy
            loss = loss_function(prediction, y)
            losses.append(loss)

            # Cohen's Kappa
            predictions_np = predicted_classes.detach().cpu().numpy()
            labels_np = y.detach().cpu().numpy()
            kappa = cohen_kappa_score(predictions_np, labels_np)
            kappas.append(kappa)

            # Matthew's Correlation Coefficient
            mcc = matthews_corrcoef(predictions_np, labels_np)
            mcc_list.append(mcc)

            # Confusion matrix
            labels = list(range(0, 6))
            conf_matrix_batch = confusion_matrix(y_pred=predictions_np, y_true=labels_np, labels=labels)
            conf_matrix_total = conf_matrix_batch + conf_matrix_total

            # Confusion Matrix 2
            if batch_number == 0:
                cm_total = ConfusionMatrix(actual_vector=labels_np, predict_vector=predictions_np,
                                           classes=[0, 1, 2, 3, 4, 5])
                batch_number += 1
            elif batch_number != 0:
                cm = ConfusionMatrix(actual_vector=labels_np, predict_vector=predictions_np, classes=[0, 1, 2, 3, 4, 5])
                cm_total = cm_total.combine(cm)
    return losses, kappas, mcc_list, conf_matrix_total, cm_total
