import numpy as np
from .tensor import Tensor


class Activation:
    def forward(self, tensor: Tensor) -> Tensor:
        raise NotImplementedError
    

class ReLU(Activation):
    def forward(self, tensor: int or float or Tensor) -> int or float or Tensor:
        if isinstance(tensor, Tensor):
            return Tensor(np.maximum(tensor.data, 0))
        else:
            return max(0, tensor)



class Sigmoid(Activation):
    def forward(self, tensor: int or float or Tensor) -> int or float or Tensor:
        if isinstance(tensor, Tensor):
            return Tensor(1 / (1 + np.exp(-tensor.data)))
        else:
            return 1 / (1 + np.exp(-tensor))




class Tanh(Activation):
    def forward(self, tensor: int or float or Tensor) -> int or float or Tensor:
        if isinstance(tensor, Tensor):
            return Tensor(np.tanh(tensor.data))
        else:
            return np.tanh(tensor)



class Softmax(Activation):
    def forward(self, tensor: int or float or Tensor) -> int or float or Tensor:
        if isinstance(tensor, Tensor):                            
            exp_data = np.exp(tensor.data - np.max(tensor.data)) 
            sum_exp = np.sum(exp_data, axis=-1, keepdims=True)
            return Tensor(exp_data / sum_exp)
        else:
            exp_data = np.exp(tensor - np.max(tensor)) 
            sum_exp = np.sum(exp_data, axis=-1, keepdims=True)
            return exp_data / sum_exp



activation_dict = {
    'relu': ReLU(),
    'sigmoid': Sigmoid(),
    'tanh': Tanh(),
    'softmax': Softmax(),
}




def activate(activation_string: str, tensor: Tensor) -> Tensor:
    if activation_string not in activation_dict:
        raise ValueError(f"Invalid activation string: {activation_string}.\n Please see documentation to see implemented activation functions")
    activation_strategy = activation_dict[activation_string]
    return activation_strategy.forward(tensor)
