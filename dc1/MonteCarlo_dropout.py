import torch
from pathlib import Path
from net import Net
from torch.utils.data import DataLoader
from image_dataset import ImageDataset

def enable_dropout(model):
    """Function to enable dropout layers during inference."""
    model.train()
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.train()

def monte_carlo_predict(model, input_tensor, n_samples=100):
    """Perform Monte Carlo predictions with enabled dropout."""
    predictions = []
    for _ in range(n_samples):
        predictions.append(model(input_tensor))
    predictions = torch.stack(predictions)
    return predictions.mean(dim=0), predictions.std(dim=0)

model = Net(n_classes=6)

enable_dropout(model)

image_dataset = ImageDataset(Path("data/X_train.npy"), Path("data/Y_train.npy"))

data_loader = DataLoader(image_dataset, batch_size=32, shuffle=False)  # Set your batch_size as required

inputs, _ = next(iter(data_loader))

mean_predictions, prediction_stddev = monte_carlo_predict(model, inputs, n_samples=100)

mean_np = mean_predictions.detach().cpu().numpy()
stddev_np = prediction_stddev.detach().cpu().numpy()

all_mean = mean_np.mean(axis=0)
all_stddev = stddev_np.mean(axis=0)

ci_lower = mean_np - 1.96 * stddev_np
ci_upper = mean_np + 1.96 * stddev_np
confidence_interval = (ci_lower, ci_upper)



