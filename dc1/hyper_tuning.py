# Custom imports
from pathlib import Path

# Other imports
import matplotlib.pyplot as plt  # type: ignore
import plotext  # type: ignore
# Torch imports
import torch
import torch.optim as optim
from sklearn.model_selection import GridSearchCV
from skorch import NeuralNetClassifier
from torchsummary import summary  # type: ignore

from dc1.image_dataset import ImageDataset
from dc1.net import Net

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_dataset = ImageDataset(Path("data/X_train.npy"), Path("data/Y_train.npy"))
test_dataset = ImageDataset(Path("data/X_test.npy"), Path("data/Y_test.npy"))

# Load the Neural Net. NOTE: set number of distinct labels here
model = NeuralNetClassifier(module=Net,
                            module__n_classes=6,
                            device=device)
model.initialize()
print(model.device)
param_grid = {
    'batch_size': [60, 80, 100],
    'max_epochs': [10, 50, 100],
    # 'optimizer': [optim.SGD, optim.RMSprop, optim.Adagrad, optim.Adadelta,
    #               optim.Adam, optim.Adamax, optim.NAdam],
}
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=2, cv=3)
grid_result = grid.fit(torch.from_numpy(train_dataset.imgs[0:1000]).float(),
                       torch.tensor(train_dataset.targets[0:1000]).long())
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
