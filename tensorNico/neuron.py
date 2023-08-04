from .activations import sigmoid
import numpy as np

class Neuron:
    def __init__(self, weights, bias, activation_function=sigmoid):
        self.weights = weights
        self.bias = bias
        self.activation_function = activation_function

    def forward(self, inputs):
        weighted_sum = (inputs * self.weights).sum() + self.bias
        activation = self.activation_function(weighted_sum)
        return activation
