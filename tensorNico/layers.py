from .tensor import Tensor
import numpy as np

class Dense:

    def __init__(self, nneurons: int, activation: str = 'relu', use_bias: bool = True):
        self.nneurons = nneurons
        self.activation = activation if activation else 'linear'
        self.use_bias = use_bias

        # Initialize weights and biases
        self.weights = np.random.randn(self.input_shape[1], self.nneurons) * 0.01
        self.bias = np.zeros(self.nneurons)

    def __call__(self, x: Tensor) -> Tensor:
        self.input_shape = x.shape

        # Matrix multiplication and add bias
        z = np.dot(x, self.weights) + self.bias

        # Apply activation function
        if self.activation == 'relu':
            return Tensor(np.maximum(0, z))
        elif self.activation == 'sigmoid':
            return Tensor(1 / (1 + np.exp(-z)))
        else:
            return Tensor(z)
        
    def backward(self, grad: 'Tensor') -> 'Tensor':
        pass