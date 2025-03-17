import numpy as np

class FFNN():
    def __init__(self, hid_layers, size_hid_layer, weight_init, activation_func, output_activation_function="softmax"):
        self.hidden_layers= hid_layers
        self.size_hid_layer = size_hid_layer 
        self.activation_function= activation_func
        self.output_activation_function= output_activation_function
        self.weights=[]
        self.biases=[]
        self.initialize_weights_and_biases( input_size=784, output_size=10, method=weight_init)


    #-------initialize weights and bias--------
    def initialize_weights_and_biases(self, input_size, output_size, method="random"):
        if method == "random":
            self.weights.append(np.random.randn(input_size, self.size_hid_layer))
            for _ in range(self.hidden_layers - 1):
                self.weights.append(np.random.randn(self.size_hid_layer, self.size_hid_layer))
            self.weights.append(np.random.randn(self.size_hid_layer, output_size))
        elif method == "xavier":
            for i in range(len(self.weights)):
                self.weights[i] = self.weights[i] * np.sqrt(1 / self.weights[i].shape[0])
        else:
            raise ValueError("Invalid method. Choose 'random' or 'xavier'.")

        #biases
        for _ in range(self.hidden_layers):
            self.biases.append(np.zeros(self.size_hid_layer))
        self.biases.append(np.zeros(output_size))


    

    #----------activation function-------------
    def activation_func(self, x):
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
            self.post_activation.append(self.activation_func(self.pre_activation[-1]))
    
        # Compute pre-activation for output layer
        z_output = np.matmul(self.post_activation[-1], self.weights[-1]) + self.biases[-1]
        self.pre_activation.append(z_output)
    
        # Apply output activation function (which might be different from hidden layer activation)
        # For example, softmax for classification or linear for regression
        self.post_activation.append(self.output_activation(self.pre_activation[-1]))

        # Return the final output of the network
        return self.post_activation[-1]
