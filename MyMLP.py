"""
This part is to implement the MLP from scratch. I refer to the two articles as mentioned in assignment requirements.
1. https://www.freecodecamp.org/news/building-a-neural-network-from-scratch/
2. https://towardsdatascience.com/step-by-step-guide-to-building-your-own-neural-network-from-scratch-df64b1c5ab6e


As to loss/cost function, the articles use Cross-Entropy Loss. Here, Mean Squared Error is used as required by the assignment.

np.dot() is used to calculate the dot product of two arrays in the articles. here @ is used as it's more readable. @ is a syntactic sugar for np.dot().

sigmoid() is used as activation function in the articles. Here, on top of sigmoid(), ReLU() is also required by the assignment.

class variables a_vals and z_vals are used to store the values of activations and linear hypothesis respectively. By using a_vals and z_vals, I don't need to pass caches around as in the freecodecamp article. This modification makes the code structure different from the articles, but more readable to me. I don't fancy passing caches around.
"""
import numpy as np


# Activation functions and their derivatives
def relu(x):
    return np.maximum(0, x)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def relu_derivative(x):
    return (x > 0).astype(float)


def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

# both articles are using cross-entropy loss, here MSE is required by the task
# Mean Squared Error Loss and its derivative
# MSE = 1/n * sum((y_true - y_pred)^2)
def mse_loss(y_true, y_pred):
    return ((y_true - y_pred) ** 2).mean()


def mse_loss_derivative(y_true, y_pred):
    return 2 * (y_pred - y_true) / y_true.size


# Basic Multilayer Perceptron (MLP) class

class MLP:
    def __init__(self, hidden_layer_tuple, activation_layer_tuple):
        self.layers = hidden_layer_tuple
        self.activations = activation_layer_tuple
        self.learning_rate = 0.01
        self.weights = []
        self.biases = []
        

		# a_vals represents the activations of each layer
        # z_vals represents the linear hypothesis of each layer,the weighted sum of inputs at each layer
        # by using a_vals and z_vals, I don't need to pass caches around as in the freecodecamp article
        self.a_vals = []
        self.z_vals = []
        # Weight and bias initialization, refer to https://www.freecodecamp.org/news/building-a-neural-network-from-scratch/
        np.random.seed(3)
        for i in range(len(self.layers) - 1):
            self.weights.append(np.random.randn(self.layers[i], self.layers[i + 1]) * 0.01)
            self.biases.append(np.random.randn(self.layers[i + 1]))


    def forward(self, X):
        self.a_vals = [X]
        self.z_vals = [X]

        for i in range(len(self.weights)):
            # https://www.freecodecamp.org/news/building-a-neural-network-from-scratch/
            z = self.a_vals[-1] @ self.weights[i] + self.biases[i]

            self.z_vals.append(z)
            if self.activations[i] == 'relu':
                a = relu(z)
            elif self.activations[i] == 'sigmoid':
                a = sigmoid(z)
            else:  # No activation, linear
                a = z
            self.a_vals.append(a)

        return self.a_vals[-1]

    def backprop(self, y):
        # Calculate the error at the output
        dA = mse_loss_derivative(y, self.a_vals[-1])
        for i in reversed(range(len(self.weights))):
            derivative = 1
            if self.activations[i] == 'relu':
                derivative = relu_derivative(self.z_vals[i + 1])
            elif self.activations[i] == 'sigmoid':
                derivative = sigmoid_derivative(self.z_vals[i + 1])
            dZ = dA * (derivative)
            # https://www.freecodecamp.org/news/building-a-neural-network-from-scratch/
            # same logic as np.dot() used in forward()
            # the freecodecamp article is using np.dot() which works the same as @
            # @ is just a syntactic sugar for np.dot()
            dA = dZ @ self.weights[i].T
            # Weight and bias updates
            self.weights[i] -= self.z_vals[i].T @ dZ * self.learning_rate
            self.biases[i] -= np.sum(dZ, axis=0) * self.learning_rate


# Example usage:
mlp = MLP((11, 5, 1), ('relu', 'none'))  
X = np.random.randn(10, 11)  
y = np.random.randn(10, 1)  
preds = mlp.forward(X)  
mlp.backprop(y)  
