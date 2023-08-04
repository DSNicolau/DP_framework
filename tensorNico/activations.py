import numpy as np
from .tensor import Tensor

def relu(self):
    if isinstance(self, Tensor):
        return Tensor(np.maximum(self.data, 0))
    else:
        return max(0, self)
    
def sigmoid(self):
    if isinstance(self, Tensor):
        return Tensor(1 / (1 + np.exp(-self.data)))
    else:
        return 1 / (1 + np.exp(-self))

def tanh(self):
    if isinstance(self, Tensor):
        return Tensor(np.tanh(self.data))
    else:
        return np.tanh(self)
    

def leaky_relu(self, alpha=0.01):
    if isinstance(self, Tensor):
        return Tensor(np.where(self.data > 0, self.data, alpha * self.data))
    else:
        return np.where(self > 0, self, alpha * self)

def elu(self, alpha=1.0):
    if isinstance(self, Tensor):                                                                          
        return Tensor(np.where(self.data > 0, self.data, alpha * (np.exp(self.data) - 1)))
    else:
        return np.where(self > 0, self, alpha * (np.exp(self) - 1))

def softmax(self):
    if isinstance(self, Tensor):                            
        exp_data = np.exp(self.data - np.max(self.data)) 
        sum_exp = np.sum(exp_data, axis=-1, keepdims=True)
        return Tensor(exp_data / sum_exp)
    else:
        exp_data = np.exp(self - np.max(self)) 
        sum_exp = np.sum(exp_data, axis=-1, keepdims=True)
        return exp_data / sum_exp

def swish(self, beta=1.0):
    if isinstance(self, Tensor):
        return Tensor(self.data * sigmoid(beta * self.data))
    else:
        return self * sigmoid(beta * self)
