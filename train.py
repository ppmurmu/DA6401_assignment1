import numpy as np
from sklearn.model_selection import train_test_split
from keras.datasets import fashion_mnist, mnist
from ffnn import FFNN
from backpropagation import Backpropagation
from optimizers import Optimizers
import wandb
from config import sweep_configuration
import argparse



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





def sweep():
    # Initialize a Weights & Biases run
    run = wandb.init()
    parameters = wandb.config

    #set the run name with hyperparameters
    run.name = f"hl={parameters['hidden_layers']}_layerSize={parameters['neurons']}_bs={parameters['batch_size']}_ac_{parameters['activation']}_lr={parameters['learning_rate']}_opt={parameters['optimizer']}_loss={parameters['loss']}"

    #load the training dataset
    x_train, y_train = load_data('train', dataset=parameters['dataset'])    
    
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
    
    optimizer = Optimizers(nn=nn,
                    bp=bp,
                    lr=parameters['learning_rate'],
                    optimizer=parameters['optimizer'],
                    decay=parameters['decay'])

    
    
    batch_size = parameters['batch_size']
    
    # Split training data into training and validation sets (90% train, 10% validation)
    x_train_act, x_val, y_train_act, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=42)


    #-----epochs start here-----training loop------
    for epoch in range(epochs):
        for i in range(0, x_train_act.shape[0], batch_size):
            x_batch = x_train_act[i:i+batch_size]
            y_batch = y_train_act[i:i+batch_size]

            y_pred = nn.forward_propagation(x_batch)
            d_weights, d_biases = bp.backward(y_batch, y_pred)
            optimizer.run(d_weights, d_biases)
        
        optimizer.t += 1
        
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
    print(f"validation Accuracy: {val_accuracy:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")

    #log test results to wandb
    wandb.log({
        "test_loss": test_loss,
        "test_accuracy": test_accuracy
    })
    
    return nn


# Function to parse command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Train a Feedforward Neural Network on MNIST/Fashion-MNIST")
    
    parser.add_argument("-wp", "--wandb_project", type=str, default="DA6401-A1", help="Weights & Biases project name")
    parser.add_argument("-we", "--wandb_entity", type=str, default="cs24m033-iit-madras", help="Weights & Biases entity")
    parser.add_argument("-d", "--dataset", type=str, choices=["mnist", "fashion_mnist"], default="fashion_mnist", help="Dataset selection")
    parser.add_argument("-e", "--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("-b", "--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("-l", "--loss", type=str, choices=["mse", "ce"], default="ce", help="Loss function")
    parser.add_argument("-o", "--optimizer", type=str, choices=["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"], default="adam", help="Optimizer")
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("-m", "--momentum", type=float, default=0.9, help="Momentum for applicable optimizers")
    parser.add_argument("-beta", "--beta", type=float, default=0.9, help="Beta for RMSProp optimizer")
    parser.add_argument("-beta1", "--beta1", type=float, default=0.9, help="Beta1 for Adam/Nadam optimizers")
    parser.add_argument("-beta2", "--beta2", type=float, default=0.999, help="Beta2 for Adam/Nadam optimizers")
    parser.add_argument("-eps", "--epsilon", type=float, default=1e-8, help="Epsilon value for numerical stability")
    parser.add_argument("-w_d", "--weight_decay", type=float, default=0.0, help="Weight decay for optimizers")
    parser.add_argument("-w_i", "--weight_init", type=str, choices=["random", "xavier"], default="xavier", help="Weight initialization method")
    parser.add_argument("-nhl", "--num_layers", type=int, default=4, help="Number of hidden layers")
    parser.add_argument("-sz", "--hidden_size", type=int, default=128, help="Number of neurons in each hidden layer")
    parser.add_argument("-a", "--activation", type=str, choices=["identity", "sigmoid", "tanh", "ReLU"], default="tanh", help="Activation function")
    
    return parser.parse_args()

def create_sweep_config(args):
    return {
        "method": "random",
        "name": "sweep",
        "metric": {
            "goal": "maximize",
            "name": "val_accuracy"
        },
        "parameters": {
            "batch_size": {"value": args.batch_size},
            "learning_rate": {"value": args.learning_rate},
            "neurons": {"value": args.hidden_size},
            "hidden_layers": {"value": args.num_layers},
            "activation": {"value": args.activation},
            "weight_init": {"value": args.weight_init},
            "optimizer": {"value": args.optimizer},
            "loss": {"value": args.loss},
            "epochs": {"value": args.epochs},
            "decay": {"value": args.weight_decay},
            "dataset": {"value": args.dataset}
        }
    }

#-----------wandb stuff-------------------
wandb.login()

#----store arg parser----
args = parse_args()
sweep_config = create_sweep_config(args)  

#you can change the count value in wandb agent below to higher values and use config.py file to create various configurations

#-----

wandb_id = wandb.sweep(sweep_config,entity=parse_args().wandb_entity, project=parse_args().wandb_project)

wandb.agent(wandb_id, function=sweep, count=1)

wandb.finish()

