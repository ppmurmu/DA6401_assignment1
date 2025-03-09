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


class RMSProp:
    def __init__(self, nn, lr=0.001, beta=0.9, epsilon=1e-8, decay=0.0):
        self.nn = nn
        self.lr = lr
        self.beta = beta
        self.epsilon = epsilon 
        self.decay = decay

        # Initialize gradients
        self.h_weights = [np.zeros_like(w) for w in self.nn.weights]
        self.h_biases = [np.zeros_like(b) for b in self.nn.biases]

    def step(self, d_weights, d_biases):
        learning_rate = self.lr
        weight_decay = self.decay
        
        for i in range(self.nn.hidden_layers + 1):
            self.h_weights[i] = self.beta * self.h_weights[i] + (1 - self.beta) * d_weights[i]**2
            self.h_biases[i] = self.beta * self.h_biases[i] + (1 - self.beta) * d_biases[i]**2

            # using RMSProp formula
            self.nn.weights[i] -= (learning_rate / (np.sqrt(self.h_weights[i]) + self.epsilon)) * d_weights[i] + weight_decay * self.nn.weights[i] * learning_rate
            self.nn.biases[i] -= (learning_rate / (np.sqrt(self.h_biases[i]) + self.epsilon)) * d_biases[i] + weight_decay * self.nn.biases[i] * learning_rate

#---------Adam-------------
class Adam:
    def __init__(self, nn, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, decay=0.0):
        self.nn = nn
        self.lr = lr
        self.beta1 = beta1  #decay rate for first moment
        self.beta2 = beta2  #decay rate for second moment
        self.epsilon = epsilon  
        self.decay = decay
        self.t = 0  #time step

        # Initialize first moment (momentum)
        self.hm_weights = [np.zeros_like(w) for w in self.nn.weights]
        self.hm_biases = [np.zeros_like(b) for b in self.nn.biases]
        
        # Initialize second moment (RMSProp)
        self.h_weights = [np.zeros_like(w) for w in self.nn.weights]
        self.h_biases = [np.zeros_like(b) for b in self.nn.biases]

    def step(self, d_weights, d_biases):
        self.t += 1  # Increment time step
        learning_rate = self.lr
        weight_decay = self.decay
        
        for i in range(self.nn.hidden_layers + 1):
            # update first moment etimate - momentum
            self.hm_weights[i] = self.beta1 * self.hm_weights[i] + (1 - self.beta1) * d_weights[i]
            self.hm_biases[i] = self.beta1 * self.hm_biases[i] + (1 - self.beta1) * d_biases[i]

            # upate second moment estimate - RMSProp
            self.h_weights[i] = self.beta2 * self.h_weights[i] + (1 - self.beta2) * d_weights[i]**2
            self.h_biases[i] = self.beta2 * self.h_biases[i] + (1 - self.beta2) * d_biases[i]**2

            # compute corrected ifrst moment - momentum
            hm_weights_hat = self.hm_weights[i] / (1 - self.beta1**self.t)
            hm_biases_hat = self.hm_biases[i] / (1 - self.beta1**self.t)

            # compute corrected second moment  -RMSprop
            h_weights_hat = self.h_weights[i] / (1 - self.beta2**self.t)
            h_biases_hat = self.h_biases[i] / (1 - self.beta2**self.t)

            # using Adam formula
            self.nn.weights[i] -= learning_rate * (hm_weights_hat / (np.sqrt(h_weights_hat) + self.epsilon)) + weight_decay * self.nn.weights[i] * learning_rate
            self.nn.biases[i] -= learning_rate * (hm_biases_hat / (np.sqrt(h_biases_hat) + self.epsilon)) + weight_decay * self.nn.biases[i] * learning_rate


#-----Nadam---------
class NAdam:
    def __init__(self, nn, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, decay=0.0):
        self.nn = nn
        self.lr = lr
        self.beta1 = beta1  #decay rate for first moment
        self.beta2 = beta2  #decay rate for second moment
        self.epsilon = epsilon  # Small constant for numerical stability
        self.decay = decay
        self.t = 0  # Time step counter

        # Initialize first moment (momentum)
        self.hm_weights = [np.zeros_like(w) for w in self.nn.weights]
        self.hm_biases = [np.zeros_like(b) for b in self.nn.biases]
        
        # Initialize second moment (RMSProp)
        self.h_weights = [np.zeros_like(w) for w in self.nn.weights]
        self.h_biases = [np.zeros_like(b) for b in self.nn.biases]

    def step(self, d_weights, d_biases):
        self.t += 1  # Increment time step
        learning_rate = self.lr
        weight_decay = self.decay
        
        for i in range(self.nn.hidden_layers + 1):
            # update first moment estimate- momentum
            self.hm_weights[i] = self.beta1 * self.hm_weights[i] + (1 - self.beta1) * d_weights[i]
            self.hm_biases[i] = self.beta1 * self.hm_biases[i] + (1 - self.beta1) * d_biases[i]

            # Update second moment estimate- RMSProp 
            self.h_weights[i] = self.beta2 * self.h_weights[i] + (1 - self.beta2) * d_weights[i]**2
            self.h_biases[i] = self.beta2 * self.h_biases[i] + (1 - self.beta2) * d_biases[i]**2

            # Compute corrected first moment estimate - momentum
            hm_weights_hat = self.hm_weights[i] / (1 - self.beta1**self.t)
            hm_biases_hat = self.hm_biases[i] / (1 - self.beta1**self.t)

            # Compute corrected second estimate estimate - momentum
            h_weights_hat = self.h_weights[i] / (1 - self.beta2**self.t)
            h_biases_hat = self.h_biases[i] / (1 - self.beta2**self.t)

            # compute nesterov momentum term for NAdam
            temp_update_w = self.beta1 * hm_weights_hat + ((1 - self.beta1) / (1 - self.beta1**self.t)) * d_weights[i]
            temp_update_b = self.beta1 * hm_biases_hat + ((1 - self.beta1) / (1 - self.beta1**self.t)) * d_biases[i]

            # using NAdam formula
            self.nn.weights[i] -= learning_rate * (temp_update_w / (np.sqrt(h_weights_hat) + self.epsilon)) + weight_decay * self.nn.weights[i] * learning_rate
            self.nn.biases[i] -= learning_rate * (temp_update_b / (np.sqrt(h_biases_hat) + self.epsilon)) + weight_decay * self.nn.biases[i] * learning_rate

#------- ADD OPTIMIZER CLASSES HERE--------------------



