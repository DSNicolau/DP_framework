import numpy as np
from .tensor import Tensor


class Activation:
    def apply(self, tensor: Tensor) -> Tensor:
        raise NotImplementedError
    
def relu(self):
    if isinstance(self, Tensor):
        return Tensor(np.maximum(self.data, 0))
    else:
        return max(0, self)
    
class ReLU(Activation):
    def apply(self, tensor: int or float or Tensor) -> int or float or Tensor:
        return relu(tensor)


def sigmoid(self):
    if isinstance(self, Tensor):
        return Tensor(1 / (1 + np.exp(-self.data)))
    else:
        return 1 / (1 + np.exp(-self))

class Sigmoid(Activation):
    def apply(self, tensor: int or float or Tensor) -> int or float or Tensor:
        return sigmoid(tensor)


def tanh(self):
    if isinstance(self, Tensor):
        return Tensor(np.tanh(self.data))
    else:
        return np.tanh(self)

class Tanh(Activation):
    def apply(self, tensor: int or float or Tensor) -> int or float or Tensor:
        return tanh(tensor)


def softmax(self):
    if isinstance(self, Tensor):                            
        exp_data = np.exp(self.data - np.max(self.data)) 
        sum_exp = np.sum(exp_data, axis=-1, keepdims=True)
        return Tensor(exp_data / sum_exp)
    else:
        exp_data = np.exp(self - np.max(self)) 
        sum_exp = np.sum(exp_data, axis=-1, keepdims=True)
        return exp_data / sum_exp

class Softmax(Activation):
    def apply(self, tensor: int or float or Tensor) -> int or float or Tensor:
        return softmax(tensor)



activation_dict = {
    'relu': ReLU(),
    'sigmoid': Sigmoid(),
    'tanh': Tanh(),
    'softmax': Softmax(),
}


def activation(activation_string: str, tensor: Tensor) -> Tensor:
    if activation_string not in activation_dict:
        raise ValueError(f"Invalid activation string: {activation_string}.\n Please see documentation to see implemented activation functions")
    activation_strategy = activation_dict[activation_string]
    return activation_strategy.apply(tensor)
