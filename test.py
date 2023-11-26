import tensorNico as tn
import numpy as np
# Create Tensor objects
tensor_a = tn.Tensor([[2, 3, 4],[5,6,7]])
tensor_b = tn.Tensor([1, 2, 3])


tensor_b = tensor_a.transpose()
print(tensor_a)
print(tensor_b)
print(tensor_a==tensor_b.transpose())
print(tensor_a.sum())