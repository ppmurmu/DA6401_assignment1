import numpy as np

class FFNN():
    def __init__(self, epochs, hid_layers, size_hid_layer, w_decay, learning_rate, optimizer, batch_size, weight_init, activation_func, output_activation_function="softmax"):
        self.epochs = epochs
        self.hid_layers= hid_layers
        self.size_hid_layer = size_hid_layer 
        self.w_decay= w_decay
        self.learning_rate = learning_rate			# Learning rate
        self.optimizer= optimizer
        self.batch_size= batch_size
        self.weight_init= weight_init   #weight initializer
        self.activation_func= activation_func
        self.output_activation_function= output_activation_function
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

    #-------forward  propagation code--------------
    def forward_propagation(self, x):
        # Initialize lists to store pre-activation and post-activation values
        self.pre_activation, self.post_activation = [x], [x]
    
        # go through each hidden layer
        for i in range(self.hidden_layers):
            # Compute pre-activation: z = a_prev * W + b
            z = np.matmul(self.post_activation[-1], self.weights[i]) + self.biases[i]
            self.pre_activation.append(z)
        
            # Apply activation function to get post-activation
            self.post_activation.append(self.activation(self.pre_activation[-1]))
    
        # Compute pre-activation for output layer
        z_output = np.matmul(self.post_activation[-1], self.weights[-1]) + self.biases[-1]
        self.pre_activation.append(z_output)
    
        # Apply output activation function (which might be different from hidden layer activation)
        # For example, softmax for classification or linear for regression
        self.post_activation.append(self.output_activation(self.pre_activation[-1]))

        # Return the final output of the network
        return self.post_activation[-1]

		