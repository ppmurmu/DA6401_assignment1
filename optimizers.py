#--- this file contains all the optimizer classes-------
import numpy as np

#nn is the neural network instance we pass.


#-------- Stochastic Gradient Descent ------------------
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


#------- ADD OPTIMIZER CLASSES HERE--------------------
