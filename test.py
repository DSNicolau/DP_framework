import tensorNico as tn
import numpy as np
import tensorNico.activations_v2 as act
# Create Tensor objects
tensor_a = tn.Tensor([[2, 3, 4],[5,6,7]])
tensor_b = tn.Tensor([-1,4,9])



# print(tensor_a.reshape((3,2)).pow(tensor_b))
# print(tensor_b.__truediv__(2))
print(act.activate('ah',tensor_b))
print(act.relu(tensor_b))

activation = act.activation_dict['relu']
print(activation.apply(tensor_b))

