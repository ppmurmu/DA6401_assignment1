import numpy as np
from sklearn.model_selection import train_test_split
from keras.datasets import fashion_mnist, mnist
from ffnn import FFNN
from backpropagation import Backpropagation
from optimizers import SGD, MomentumGD, NAG, RMSProp, Adam, NAdam
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

#----choose optimizer------
def choose_optimizer(nn, 
                 bp, 
                 lr=0.001, 
                 optimizer="sgd", 
                 momentum=0.9,
                 epsilon=1e-8,
                 beta=0.9,
                 beta1=0.9,
                 beta2=0.999, 
                 t=0,
                 decay=0):
    if optimizer == "sgd":
        return SGD(nn, lr=lr, decay=decay)
    elif optimizer == "momentum":
        return MomentumGD(nn, lr=lr, momentum=momentum, decay=decay)
    elif optimizer == "nag":
        return NAG(nn, lr=lr, momentum=momentum, decay=decay)
    elif optimizer == "rmsprop":
        return RMSProp(nn, lr=lr, beta=beta, epsilon=epsilon, decay=decay)
    elif optimizer == "adam":
        adam_opt = Adam(nn, lr=lr, beta1=beta1, beta2=beta2, epsilon=epsilon, decay=decay)
        adam_opt.t = t
        return adam_opt
    elif optimizer == "nadam":
        nadam_opt = NAdam(nn, lr=lr, beta1=beta1, beta2=beta2, epsilon=epsilon, decay=decay)
        nadam_opt.t = t
        return nadam_opt
    else:
        raise Exception("Invalid optimizer")




def sweep():
    # Initialize a Weights & Biases run
    run = wandb.init()
    parameters = wandb.config

    #set the run name with hyperparameters
    run.name = f"hl={parameters['hidden_layers']}_layerSize={parameters['neurons']}_bs={parameters['batch_size']}_ac_{parameters['activation']}_lr={parameters['learning_rate']}_opt={parameters['optimizer']}_loss={parameters['loss']}"

    #load the training dataset
    x_train, y_train = load_data('train')
    
    
    epochs = parameters['epochs']
    loss_func= parameters['loss']
    

    #Initialize the neural network
    nn = FFNN(        
                         hid_layers=parameters['hidden_layers'], 
                         size_hid_layer=parameters['neurons'], 
                         activation_func=parameters['activation'], 
                         weight_init=parameters['weight_init'],)
    # Initialize the backpropagation algorithm
    bp = Backpropagation(nn=nn, 
                         loss=parameters['loss'],
                         )
    # Choose and configure the optimizer
    optimizer = choose_optimizer(nn=nn,
                    bp=bp,
                    lr=parameters['learning_rate'],
                    optimizer=parameters['optimizer'],
                    decay=parameters['decay'])

    
    
    batch_size = parameters['batch_size']
    
    # Split training data into training and validation sets (90% train, 10% validation)
    x_train_act, x_val, y_train_act, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=42)

    #print("Initial Accuracy: {}".format(np.sum(np.argmax(nn.forward_propagation(x_train), axis=1) == np.argmax(y_train, axis=1)) / y_train.shape[0]))

    #-----epochs start here-----training loop------
    for epoch in range(epochs):
        for i in range(0, x_train_act.shape[0], batch_size):
            x_batch = x_train_act[i:i+batch_size]
            y_batch = y_train_act[i:i+batch_size]

            y_pred = nn.forward_propagation(x_batch)
            d_weights, d_biases = bp.backward(y_batch, y_pred)
            optimizer.step(d_weights, d_biases)
        
        
        #loss and accuracy after each epoch
        y_pred = nn.forward_propagation(x_train_act)
        train_loss = loss(loss_func, y_train_act, y_pred)
        train_accuracy = np.sum(np.argmax(y_pred, axis=1) == np.argmax(y_train_act, axis=1)) / y_train_act.shape[0]

        #print training process
        print(f"Epoch: {epoch + 1}, Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.4f}")


        #validation loss and accuracy
        y_pred_val = nn.forward_propagation(x_val)
        val_loss = loss(loss_func, y_val, y_pred_val)
        val_accuracy = np.sum(np.argmax(y_pred_val, axis=1) == np.argmax(y_val, axis=1)) / y_val.shape[0]

        #log on wandb
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy
        })

    #-----------TESTING PHASE---------------
    x_test, y_test = load_data('test') #load test dataset

    #compute test loss and accuracy
    y_pred_test = nn.forward_propagation(x_test)
    test_loss = loss(loss_func, y_test, y_pred_test)
    test_accuracy = np.sum(np.argmax(y_pred_test, axis=1) == np.argmax(y_test, axis=1)) / y_test.shape[0]

    #printing test accuracy
    print(f"Test Accuracy: {test_accuracy:.4f}")

    #log test results to wandb
    wandb.log({
        "test_loss": test_loss,
        "test_accuracy": test_accuracy
    })
    
    return nn