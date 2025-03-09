import numpy as np

#-------- Stochastic Gradient Descent ------------------
#nn is the neural network instance we pass.
class SGD:
    def __init__(self, nn, lr=0.001, decay=0.0):
        self.nn = nn
        self.lr = lr
        self.decay = decay

    def step(self, d_weights, d_biases):
        learning_rate = self.lr
        weight_decay = self.decay  
        for layer in range(self.nn.hidden_layers + 1):
        
            # Compute weight update using learning rate and decay
            weight_update = learning_rate * (d_weights[layer] + weight_decay * self.nn.weights[layer])
            bias_update = learning_rate * (d_biases[layer] + weight_decay * self.nn.biases[layer])
        
            # Update weights and biases
            self.nn.weights[layer] -= weight_update
            self.nn.biases[layer] -= bias_update


class MomentumGD:
    def __init__(self, nn, lr=0.001, momentum=0.9, decay=0.0):
        self.nn = nn
        self.lr = lr
        self.momentum = momentum
        self.decay = decay

        # Initialize momentum terms
        self.h_weights = [np.zeros_like(w) for w in self.nn.weights]
        self.h_biases = [np.zeros_like(b) for b in self.nn.biases]

    def step(self, d_weights, d_biases):
        learning_rate = self.lr
        weight_decay = self.decay  
        for i in range(self.nn.hidden_layers + 1):
            # Compute momentum update
            self.h_weights[i] = self.momentum * self.h_weights[i] + d_weights[i]
            self.h_biases[i] = self.momentum * self.h_biases[i] + d_biases[i]

            # Update weights and biases
            self.nn.weights[i] -= learning_rate * (self.h_weights[i] + weight_decay * self.nn.weights[i])
            self.nn.biases[i] -= learning_rate * (self.h_biases[i] + weight_decay * self.nn.biases[i])


#-------- Nesterov Accelerated Gradient Descent  ------------------
class NAG:
    def __init__(self, nn, lr=0.001, momentum=0.9, decay=0.0):
        self.nn = nn
        self.lr = lr
        self.momentum = momentum
        self.decay = decay

        # Initialize momentum terms
        self.h_weights = [np.zeros_like(w) for w in self.nn.weights]
        self.h_biases = [np.zeros_like(b) for b in self.nn.biases]

    def step(self, d_weights, d_biases):
        learning_rate = self.lr
        weight_decay = self.decay  
        for i in range(self.nn.hidden_layers + 1):
            # Compute Nesterov momentum update
            self.h_weights[i] = self.momentum * self.h_weights[i] + d_weights[i]
            self.h_biases[i] = self.momentum * self.h_biases[i] + d_biases[i]

            # Update weights and biases with NAG formula
            self.nn.weights[i] -= learning_rate * (self.momentum * self.h_weights[i] + d_weights[i] + weight_decay * self.nn.weights[i])
            self.nn.biases[i] -= learning_rate * (self.momentum * self.h_biases[i] + d_biases[i] + weight_decay * self.nn.biases[i])

#------- ADD OPTIMIZER CLASSES HERE--------------------
