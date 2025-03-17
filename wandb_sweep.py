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
    
    
#-----pre processing of data-----------
def preprocess(x, y):
        x = x.reshape(x.shape[0], 784) / 255  # Flatten & normalize
        y = np.eye(10)[y]  # One-hot encode labels
        return x, y

#---function to load data--------------
def load_data(type, dataset='fashion_mnist'):

    x, y, x_test, y_test = (), (), (), ()
    
    if dataset == 'mnist':
        (x, y), (x_test, y_test) = mnist.load_data()
    elif dataset == 'fashion_mnist':
        (x, y), (x_test, y_test) = fashion_mnist.load_data()

    if type == 'train':
        return preprocess(x, y)
    elif type == 'test':
        return preprocess(x_test, y_test)
    

