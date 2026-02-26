import numpy as np
import matplotlib.pyplot as plt
from enum import Enum

class w_init(Enum):
    XAVIER = 1
    HE = 2


class Linear_Layer:

    def xavier_init(self, n_in, n_out):
        limit = np.sqrt(6 / (n_in + n_out))
        return np.random.uniform(-limit, limit, (n_out, n_in))    

    def he_init(self, n_in, n_out):
        std = np.sqrt(2 / n_in)
        return np.random.randn(n_out, n_in) * std

    def __init__(self, input_dim, output_dim, init_type=w_init.XAVIER):

        if init_type == w_init.XAVIER:
            self.weights = self.xavier_init(input_dim, output_dim)
        
        elif init_type == w_init.HE:
            self.weights = self.he_init(input_dim, output_dim)
        
        self.biases = np.zeros(output_dim)
        self.input = None
        self.grad_weights = None
        self.grad_biases = None

        self.velocity = {'weights': np.zeros_like(self.weights), 'biases': np.zeros_like(self.biases)}

    def forward(self, input):
        self.input = input
        return np.dot(input, self.weights.T) + self.biases

    def backward(self, grad_output):
        self.grad_weights = np.dot(grad_output.T, self.input)
        self.grad_biases = np.sum(grad_output, axis=0)
        return np.dot(grad_output, self.weights)

    def update(self, learning_rate, momentum=0, velocity=None):
        if not momentum:
            self.weights -= learning_rate * self.grad_weights
            self.biases -= learning_rate * self.grad_biases
        else:
            velocity['weights'] = momentum * velocity['weights'] + (1 - momentum) * self.grad_weights
            velocity['biases'] = momentum * velocity['biases'] + (1 - momentum) * self.grad_biases
            self.weights -= velocity['weights'] * learning_rate 
            self.biases -= velocity['biases'] * learning_rate


class Activation:
    def __init__(self, func, func_d):
        self.func = func
        self.func_d = func_d
        self.input = None

    def forward(self, input):
        self.input = input
        return self.func(input)

    def backward(self, grad_output):
        return grad_output * self.func_d(self.input)


def relu(x):
    return np.maximum(0, x)

def relu_d(x):
    return (x > 0).astype(float)

def sigmoid(x):
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))

def sigmoid_d(x):
    s = sigmoid(x)
    return s * (1 - s)

def tanh(x):
    return np.tanh(x)

def tanh_d(x):
    return 1 - np.tanh(x)**2


class MSE_Loss:

    def __init__(self):
        self.prediction = None
        self.target = None

    def forward(self, prediction, target):
        self.prediction = prediction
        self.target = target
        return np.mean((prediction - target) ** 2)

    def backward(self):
        return 2 * (self.prediction - self.target) / self.target.size



class Model:
    def __init__(self, layers):
        self.layers = layers

    def forward(self, input):
        output = input
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def backward(self, grad_output):
        for layer in reversed(self.layers):
            grad_output = layer.backward(grad_output)

    def update(self, learning_rate, momentum=0):
        for layer in self.layers:
            if hasattr(layer, 'update'):
                if momentum:
                    layer.update(learning_rate, momentum, layer.velocity)
                else:
                    layer.update(learning_rate)



def main():
    
    operation = input("Input which operation you want to solve (1: XOR, 2: AND, 3: OR): ")

    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

    if operation == '1':
        y = np.array([[0], [1], [1], [0]])
    elif operation == '2':
        y = np.array([[0], [0], [0], [1]])
    elif operation == '3':
        y = np.array([[0], [1], [1], [1]])
    else:
        print("Invalid input: must be one of the specified values")
        return


    layer_count = input("Input the number of hidden layers: ")

    if not layer_count.isdigit():
        print("Invalid input: must be an integer")
        return

    layer_count = int(layer_count)

    layers = []

    in_neuron_count = 2
    
    for i in range(layer_count):

        out_neuron_count = input(f"Input the number of neurons for the {i + 1}. hidden layer: ")

        if not out_neuron_count.isdigit():
            print("Invalid input: must be an integer")
            return
        
        out_neuron_count = int(out_neuron_count)

        activation = input("Input the activation function between this layer and the previous layer (1: Tanh, 2: ReLU, 3: Sigmoid): ")

        if activation == '1':
            layers.append(Linear_Layer(in_neuron_count, out_neuron_count, init_type=w_init.XAVIER))
            layers.append(Activation(tanh, tanh_d))
        elif activation == '2':
            layers.append(Linear_Layer(in_neuron_count, out_neuron_count, init_type=w_init.HE))
            layers.append(Activation(relu, relu_d))
        elif activation == '3':
            layers.append(Linear_Layer(in_neuron_count, out_neuron_count, init_type=w_init.XAVIER))
            layers.append(Activation(sigmoid, sigmoid_d))
        else:
            print("Invalid input: must be one of the specified values")
            return
        
        in_neuron_count = out_neuron_count

    layers.append(Linear_Layer(in_neuron_count, 1, init_type=w_init.XAVIER))
    layers.append(Activation(sigmoid, sigmoid_d))


    model = Model(layers)
    loss_function = MSE_Loss()

    learning_rate = input("Input learning rate: ")
    learning_rate = float(learning_rate)

    momentum = input("Input momentum: ")
    momentum = float(momentum)

    epoch_count = input("Input number of epochs: ")

    if not epoch_count.isdigit():
        print("Invalid input: must be an integer")
        return
    
    epoch_count = int(epoch_count)

    losses = []

    for epoch in range(epoch_count):
        predictions = model.forward(X)
        loss = loss_function.forward(predictions, y)
        losses.append(loss)

        grad_output = loss_function.backward()
        model.backward(grad_output)

        model.update(learning_rate, momentum=momentum)

        print(f"Epoch {epoch}, Loss: {loss:.4f}")

    print("Predictions for a trained network: ")
    print(model.forward(X))


    plt.plot(losses)
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()



if __name__ == "__main__":
    main()