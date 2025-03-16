#----backpropagation class----------
class Backpropagation():

    #initialize values
    def __init__(self, 
                 nn,
                 loss="ce"):
        
        self.nn=nn
        self.loss= loss
    
    #--------derivative of activation func as used FFNN--------------
    def activation_derivative(self, x):
        # x is the post-activation value
        if self.nn.activation_function == "sigmoid":
            return x * (1 - x)
            
        elif self.nn.activation_function == "relu":
            return np.where(x > 0, 1, 0) #(x > 0).astype(int)
            
        elif self.nn.activation_function == "tanh":
            return 1 - x ** 2
            
        elif self.nn.activation_function == "identity":
            return np.ones(x.shape)
        else:
            raise Exception("Invalid activation function")

    #--------derivative of output activation func--------------
    def output_activation_derivative(self, y, y_pred):
        if self.nn.output_activation_function == "softmax":
            return np.diag(y_pred) - np.outer(y_pred, y_pred)
        else:
            raise Exception("Invalid output activation function")

    def backward(self, y, y_pred):

        #initialize
        self.d_h, self.d_a, self.d_weights, self.d_biases = [], [], [], []
        

        #----derivatiive of loss function---------
        if self.loss == "ce":
            self.d_h.append(-y / y_pred)
        elif self.loss == "mse":
            self.d_h.append(y_pred - y)
        else:
            raise Exception("Invalid loss function")

        output_derivative_matrix = []
        for i in range(y_pred.shape[0]):
            loss_grad = self.d_h[-1][i]  # Loss derivative
            softmax_grad = self.output_activation_derivative(y[i], y_pred[i])  # Softmax derivative
            output_derivative_matrix.append(np.matmul(loss_grad, softmax_grad))  # Chain rule application
            
        self.d_a.append(np.array(output_derivative_matrix)) # Store activation derivative

        # Backpropagation loop (from output layer to first hidden layer)
        for layer in range(self.nn.hidden_layers, 0, -1):
            # Compute weight gradient: dW = (post-activation from previous layer) * dA
            dW = np.matmul(self.nn.post_activation[layer].T, self.d_a[-1])
            self.d_weights.append(dW)
            
            # Compute bias gradient: sum over batch dimension
            db = np.sum(self.d_a[-1], axis=0)
            self.d_biases.append(db)

            # Compute derivative of previous layer's activation (dH)
            dH = np.matmul(self.d_a[-1], self.nn.weights[layer].T)
            self.d_h.append(dH)

            # Compute derivative of activation function
            dA_prev = dH * self.activation_derivative(self.nn.post_activation[layer])
            self.d_a.append(dA_prev)

        # Compute weight and bias gradients for the first layer
        dW_first = np.matmul(self.nn.post_activation[0].T, self.d_a[-1])
        self.d_weights.append(dW_first)

        db_first = np.sum(self.d_a[-1], axis=0)
        self.d_biases.append(db_first)


        # Reverse lists to match correct order (first layer first, output layer last)
        self.d_weights.reverse()
        self.d_biases.reverse()

        # Normalize gradients by batch size to avoid exploding values
        batch_size = y.shape[0]
        for i in range(len(self.d_weights)):
            self.d_weights[i] /= batch_size
            self.d_biases[i] /= batch_size

        return self.d_weights, self.d_biases