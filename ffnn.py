import numpy as np

class FFNN():
    def __init__(self, epochs, hid_layers, size_hid_layer, w_decay, learning_rate, optimizer, batch_size, weight_init, activation_func):
        self.epochs = epochs
        self.hid_layers= hid_layers
        self.size_hid_layer = size_hid_layer 
        self.w_decay= w_decay
        self.learning_rate = learning_rate			# Learning rate
        self.optimizer= optimizer
        self.batch_size= batch_size
        self.weight_init= weight_init   #weight initializer
        self.activation_func= activation_func
        self.weights=[]
        self.bias=[]


    #-------initialize weights and bias--------
    def initialize_weights_and_bias(self, input_size, output_size, method="random"):
        if method == "random":
            weights = np.random.randn(input_size, output_size)  # Standard normal distribution
        elif method == "xavier":
            scale = np.sqrt(2 / (input_size + output_size))  # Xavier (Glorot) Initialization
            weights = np.random.randn(input_size, output_size) * scale
        else:
            raise ValueError("Invalid method. Choose 'random' or 'xavier'.")
        bias = np.zeros(output_size)
        return weights, bias


    

    #----------activation function-------------
    def activation_funcion(self, x):
        if self.activation_function == "sigmoid":
            return 1 / (1 + np.exp(-x))
        elif self.activation_function == "tanh":
            return np.tanh(x)
        elif self.activation_function == "relu":
            return np.maximum(0, x)
        else:
            raise Exception("Invalid activation function")

    #---------output activation function------------------
    
    def output_activation(self, x):
        if self.output_activation_function == "softmax":
            exp_x = np.exp(x - np.max(x)) 
            return exp_x / np.sum(exp_x, axis=0)
        else:
            raise Exception("Invalid output activation function")