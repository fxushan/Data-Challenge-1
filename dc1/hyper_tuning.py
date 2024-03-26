# Custom imports
from pathlib import Path

# Other imports
import matplotlib.pyplot as plt
import numpy as np
# Plot install
import pandas as pd
import plotext  # type: ignore
import seaborn as sns
# Torch imports
import torch
from skopt import BayesSearchCV
from skopt.space import Integer, Real
from skorch import NeuralNetClassifier
from torchsummary import summary  # type: ignore

from dc1.image_dataset import ImageDataset
from dc1.net import Net
import random


def parameter_over_iterations(model_result):
    '''
    This function is generating a subplots with the hyperparameter values for each iteration and the overall performance score.
    The performance score is the difference between the best performing model and the worst performing model

    model_result: CV object
    '''
    param_list = list(model_result.cv_results_['params'][0].keys())
    max_col_plot = 2
    row_plot = int(np.ceil((len(param_list) + 1) / max_col_plot))
    fig, axs = plt.subplots(nrows=row_plot, ncols=np.min((max_col_plot, (len(param_list) + 1))), figsize=(30, 12))
    for i, ax in enumerate(axs.flatten()):
        if i == len(param_list):
            break
        par = param_list[i]
        param_val = list()
        for par_dict in model_result.cv_results_['params']:
            param_val.append(par_dict[par])
        sns.barplot(y=param_val, x=np.arange(len(param_val)), ax=ax)
        ax.set_title(par)
    dt = pd.DataFrame({key: val for key, val in model_result.cv_results_.items() if key.startswith('split')})
    mean_metric = dt.mean(axis=1)
    sns.barplot(y=(mean_metric.values + abs(np.min(mean_metric.values))), x=np.arange(len(mean_metric)),
                ax=axs.flatten()[i])
    axs.flatten()[i].set_title('overall metric')
    plt.show()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = 'cpu'
train_dataset = ImageDataset(Path("data/X_train.npy"), Path("data/Y_train.npy"))
test_dataset = ImageDataset(Path("data/X_test.npy"), Path("data/Y_test.npy"))
X_train = torch.from_numpy(train_dataset.imgs).float()
y_train = torch.tensor(train_dataset.targets).long()

random.seed(42)
X_sample, y_sample = zip(*random.sample(list(zip(X_train, y_train)), 2000))
X_sample = np.asarray(X_sample)
y_sample = np.asarray(y_sample)
# print(X_sample, y_sample)

# Load the Neural Net. NOTE: set number of distinct labels here
model = NeuralNetClassifier(module=Net,
                            module__n_classes=6,
                            device=device)
model.initialize()
print(model.device)

# param_grid = {
#     'batch_size': [10, 20, 40, 60, 80, 100],a
#     'max_epochs': [10, 50, 100],
#     # 'optimizer': [optim.SGD, optim.RMSprop, optim.Adagrad, optim.Adadelta,
#     #               optim.Adam, optim.Adamax, optim.NAdam],
# }
# grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=2, cv=3)
# grid_result = grid.fit(X_train, y_train)
# print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
# means = grid_result.cv_results_['mean_test_score']
# stds = grid_result.cv_results_['std_test_score']
# params = grid_result.cv_results_['params']
# for mean, stdev, param in zip(means, stds, params):
#     print("%f (%f) with: %r" % (mean, stdev, param))
# print(model.get_params())
optimizer_kwargs = {'acq_func_kwargs': {"xi": 10, "kappa": 10}}
# space = {'batch_size': Integer(10, 100),
#          # 'lr': Real(0.01, 0.55, "uniform"),
#          'max_epochs': (Integer(10, 100))}
# space = {'module__slope_1': Real(0.001, 1),
#          'module__slope_2': Real(0.001, 1),
#          'module__slope_3': Real(0.001, 1)}
space = {'module__alpha_1': Real(0.01, 2),
         'module__alpha_2': Real(0.01, 2),
         'module__alpha_3': Real(0.01, 2)}
bsearch = BayesSearchCV(estimator=model,
                        search_spaces=space, scoring='neg_mean_absolute_error', n_jobs=2, n_iter=42, cv=3,
                        optimizer_kwargs=optimizer_kwargs)
bayes_result = bsearch.fit(X_sample, y_sample)
print("Best: %f using %s" % (bayes_result.best_score_, bayes_result.best_params_))
means = bayes_result.cv_results_['mean_test_score']
stds = bayes_result.cv_results_['std_test_score']
params = bayes_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
print(model.get_params())

parameter_over_iterations(bsearch)
