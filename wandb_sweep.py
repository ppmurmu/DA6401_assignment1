import numpy as np
from sklearn.model_selection import train_test_split
from keras.datasets import fashion_mnist, mnist
import wandb

def loss(loss, y, y_pred):
    if loss == "ce": # Cross Entropy
        return -np.sum(y * np.log(y_pred))
    elif loss == "mse": # Mean Squared Error
        return np.sum((y - y_pred) ** 2) / 2
    else:
        raise Exception("Invalid loss function")