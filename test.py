import tensorNico as tn
import numpy as np
# Create Tensor objects
tensor_a = tn.Tensor([[2, 3, 4],[5,6,7]])
tensor_b = tn.Tensor([1, 2])



# print(tensor_a.reshape((3,2)).pow(tensor_b))
print(tensor_a.pow(0.5))