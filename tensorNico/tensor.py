import numpy as np

class Tensor:
    """
    A custom tensor class that provides basic tensor operations and functions, built on top of NumPy.

    This class allows for easy manipulation of tensors with various mathematical operations and element-wise functions.
    It supports addition, subtraction, multiplication, division, as well as activation functions like ReLU, sigmoid,
    and hyperbolic tangent (tanh). It also provides methods for element-wise exponentiation and logarithm.

    The tensor data can be initialized with a list or a NumPy ndarray, and the data type can be specified (default is float32).
    
    Args:
        data (list or np.ndarray): The data of the tensor.
        dtype (numpy data type, optional): The data type of the tensor (default is np.float32).

    Attributes:
        data (np.ndarray): The underlying data of the tensor.
        shape (tuple): The shape of the tensor.
        dtype (numpy data type): The data type of the tensor.

    Methods:
        __getitem__(indices): Allows indexing and slicing of the tensor.
        __setitem__(indices, value): Allows modifying tensor values using indexing.
        __add__(other): Enables element-wise addition of tensors or scalar values.
        __sub__(other): Enables element-wise subtraction of tensors or scalar values.
        __mul__(other): Enables element-wise multiplication of tensors or scalar values.
        __truediv__(other): Enables element-wise division of tensors or scalar values.
        relu(): Applies the Rectified Linear Unit (ReLU) activation function element-wise.
        sigmoid(): Applies the sigmoid activation function element-wise.
        tanh(): Applies the hyperbolic tangent (tanh) activation function element-wise.
        exp(): Computes the element-wise exponentiation of the tensor.
        log(): Computes the element-wise natural logarithm of the tensor.
        __eq__(other): Performs element-wise equality comparison with another tensor or value.
        __lt__(other): Performs element-wise less-than comparison with another tensor or value.
        __le__(other): Performs element-wise less-than or equal-to comparison with another tensor or value.
        __gt__(other): Performs element-wise greater-than comparison with another tensor or value.
        __ge__(other): Performs element-wise greater-than or equal-to comparison with another tensor or value.
        __str__(): Returns a string representation of the tensor.

    Note:
        - The class is built on top of NumPy arrays, so it benefits from NumPy's optimized operations.
        - The class supports broadcasting for element-wise operations.
        - Be cautious when using the `log` function, as it handles non-positive values gracefully but may produce
          unexpected results due to the logarithm of zero or negative values.
    """
    def __init__(self: 'Tensor', data: list or np.ndarray, dtype: np.dtype = np.float32)-> None:
        """
        Initialize a new Tensor instance.

        Args:
            data (list or np.ndarray): The data of the tensor.
            dtype (numpy data type, optional): The data type of the tensor (default is np.float32).
        """
        if isinstance(data, list):
            data = np.array(data, dtype=dtype)
        elif isinstance(data, np.ndarray):
            data = data.astype(dtype)
        else:
            raise ValueError("Unsupported data type")

        self.data = data
        self.shape = data.shape
        self.dtype = data.dtype

    def __getitem__(self: 'Tensor', indices: tuple) -> 'Tensor':
        """
        Enable indexing and slicing of the tensor.

        Args:
            indices (int, slice, tuple of int/slice): Indices or slices used to access elements.

        Returns:
            Tensor: New Tensor instance containing the selected elements.
        
        Note: 
            Views minimize unnecessary memory allocation and data copying and can lead to faster
            execution of operations since they operate directly on the underlying data without additional copying.
            Returning Tensor(self.data[indices], dtype=self.data.dtype) instead of self.data[indices]
            uses views, which is way more efficient since it takes advantage of Numpy's memory-efficient slicing methods.
            However, changes made to sliced tensors or copies will affect the original tensor. This is due to the nature of views.
            To prevent this behavior, a copy-on-write mechanism is consider. 
        """
        return self.data[indices]

    def __setitem__(self: 'Tensor', indices: tuple, value: int or float) -> None:
        """
        Modify tensor values using indexing.

        Args:
            indices (int, slice, tuple of int/slice): Indices or slices indicating where to assign values.
            value: Value to assign to the specified indices.
        """
        self.data[indices] = value

    def _broadcastable(self: 'Tensor', other: 'Tensor') -> tuple[int, ...]: 
        self_shape = self.shape
        other_shape = other.shape

        max_dims = max(len(self_shape), len(other_shape))
        self_pad = max_dims - len(self_shape)
        other_pad = max_dims - len(other_shape)

        self_shape = (1,) * self_pad + self_shape
        other_shape = (1,) * other_pad + other_shape

        broadcastable_shape = []

        for s_dim, o_dim in zip(self_shape, other_shape):
            if s_dim == 1 or o_dim == 1 or s_dim == o_dim:
                broadcastable_shape.append(max(s_dim, o_dim))
            else:
                return None

        return tuple(broadcastable_shape)
    
    def _broadcast(self: 'Tensor', other: 'Tensor') -> tuple['Tensor', 'Tensor']:
        broadcast_shape = self._broadcastable(other)

        if broadcast_shape is None:
            raise ValueError("Broadcasting dimensions are not compatible.")

        self_broadcast = self.data
        other_broadcast = other.data

        for i in range(len(self.shape)):
            if self.shape[i] < broadcast_shape[i]:
                self_broadcast = np.expand_dims(self_broadcast, axis=i)
            if other.shape[i] < broadcast_shape[i]:
                other_broadcast = np.expand_dims(other_broadcast, axis=i)

        return self_broadcast, other_broadcast



    def __add__(self: 'Tensor', other: 'Tensor') -> 'Tensor':
        if isinstance(other, Tensor):
            self_broadcast, other_broadcast = self._broadcast(other)
            return Tensor(self_broadcast + other_broadcast)
        else:
            return Tensor(self.data + other)
    
    def sum(self: 'Tensor') -> int or float:
        return np.sum(self.data)

    def __sub__(self: 'Tensor', other: 'Tensor') -> 'Tensor':
        if isinstance(other, Tensor):
            self_broadcast, other_broadcast = self._broadcast(other)
            return Tensor(self_broadcast - other_broadcast)
        else:
            return Tensor(self.data - other)

    def __mul__(self: 'Tensor', other: 'Tensor') -> 'Tensor':
        if isinstance(other, Tensor):
            self_broadcast, other_broadcast = self._broadcast(other)
            return Tensor(self_broadcast * other_broadcast)
        else:
            return Tensor(self.data * other)

    def __truediv__(self: 'Tensor', other: 'Tensor') -> 'Tensor':
        if isinstance(other, Tensor):
            self_broadcast, other_broadcast = self._broadcast(other)
            return Tensor(self_broadcast / other_broadcast)
        else:
            return Tensor(self.data / other)
    
    def exp(self: 'Tensor') -> 'Tensor':
        return Tensor(np.exp(self.data))

    def log(self: 'Tensor') -> 'Tensor':
        if np.any(self.data <= 0):
            print("Warning: Logarithm operation found non-positive values. Results may be invalid.")
        return Tensor(np.log(np.maximum(self.data, np.finfo(self.dtype).tiny)))
    
    def dot(self: 'Tensor', other: 'Tensor') -> 'Tensor':
        if isinstance(other, Tensor):
            result = np.dot(self.data, other.data)
            return Tensor(result)
        else:
            raise ValueError("Unsupported data type for matrix multiplication")
        

    def transpose(self: 'Tensor') -> 'Tensor':
        if len(self.shape) < 2:
            return self

        transposed_data = np.transpose(self.data)
        return Tensor(transposed_data, dtype=self.dtype)
    
    def reshape(self: 'Tensor', new_shape: tuple) -> 'Tensor':
        if np.prod(self.shape) != np.prod(new_shape):
            raise ValueError("Invalid reshape dimensions")

        reshaped_data = np.reshape(self.data, new_shape)
        return Tensor(reshaped_data, dtype=self.dtype)

    def __eq__(self: 'Tensor', other: any) -> 'Tensor':
        if isinstance(other, Tensor):
            self_broadcast, other_broadcast = self._broadcast(other)
            return Tensor(self_broadcast == other_broadcast)
        else:
            return Tensor(self.data == other)

    def __lt__(self: 'Tensor', other: any) -> 'Tensor':
        if isinstance(other, Tensor):
            self_broadcast, other_broadcast = self._broadcast(other)
            return Tensor(self_broadcast < other_broadcast)
        else:
            return Tensor(self.data < other)

    def __le__(self: 'Tensor', other: any) -> 'Tensor':
        if isinstance(other, Tensor):
            self_broadcast, other_broadcast = self._broadcast(other)
            return Tensor(self_broadcast <= other_broadcast)
        else:
            return Tensor(self.data <= other)

    def __gt__(self: 'Tensor', other: any) -> 'Tensor':
        if isinstance(other, Tensor):
            self_broadcast, other_broadcast = self._broadcast(other)
            return Tensor(self_broadcast > other_broadcast)
        else:
            return Tensor(self.data > other)

    def __ge__(self: 'Tensor', other: any) -> 'Tensor':
        if isinstance(other, Tensor):
            self_broadcast, other_broadcast = self._broadcast(other)
            return Tensor(self_broadcast >= other_broadcast)
        else:
            return Tensor(self.data >= other)

    def __str__(self: 'Tensor') -> str:
        return f"Tensor(data={self.data}, shape={self.shape}, dtype={self.dtype})"
