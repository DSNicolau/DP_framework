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
    def __init__(self, data, dtype=np.float32):
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

    def __getitem__(self, indices):
        """
        Enable indexing and slicing of the tensor.

        Args:
            indices (int, slice, tuple of int/slice): Indices or slices used to access elements.

        Returns:
            Tensor: New Tensor instance containing the selected elements.
        """
        return self.data[indices]
        """
        Views minimize unnecessary memory allocation and data copying and can lead to faster
        execution of operations since they operate directly on the underlying data without additional copying.
        Returning Tensor(self.data[indices], dtype=self.data.dtype) instead of self.data[indices]
        uses views, which is way more efficient since it takes advantage of Numpy's memory-efficient slicing methods.
        However, changes made to sliced tensors or copies will affect the original tensor. This is due to the nature of views.
        To prevent this behavior, a copy-on-write mechanism is consider. 
        """


    def __setitem__(self, indices, value):
        """
        Modify tensor values using indexing.

        Args:
            indices (int, slice, tuple of int/slice): Indices or slices indicating where to assign values.
            value: Value to assign to the specified indices.
        """
        self.data[indices] = value

    def __add__(self, other):
        """
        Perform element-wise addition with another tensor or scalar value.

        Args:
            other (Tensor or scalar): Another tensor or scalar to add element-wise.

        Returns:
            Tensor: New Tensor instance containing the result of the addition.
        """
        if isinstance(other, Tensor):
            return Tensor(self.data + other.data)
        else:
            return Tensor(self.data + other)

    def __sub__(self, other):
        """
        Perform element-wise subtraction with another tensor or scalar value.

        Args:
            other (Tensor or scalar): Another tensor or scalar to subtract element-wise.

        Returns:
            Tensor: New Tensor instance containing the result of the subtraction.
        """
        if isinstance(other, Tensor):
            return Tensor(self.data - other.data)
        else:
            return Tensor(self.data - other)

    def __mul__(self, other):
        """
        Perform element-wise multiplication with another tensor or scalar value.

        Args:
            other (Tensor or scalar): Another tensor or scalar to multiply element-wise.

        Returns:
            Tensor: New Tensor instance containing the result of the multiplication.
        """
        if isinstance(other, Tensor):
            return Tensor(self.data * other.data)
        else:
            return Tensor(self.data * other)

    def __truediv__(self, other):
        """
        Perform element-wise division with another tensor or scalar value.

        Args:
            other (Tensor or scalar): Another tensor or scalar to divide element-wise.

        Returns:
            Tensor: New Tensor instance containing the result of the division.
        """
        if isinstance(other, Tensor):
            return Tensor(self.data / other.data)
        else:
            return Tensor(self.data / other)

    def relu(self):
        """
        Apply the Rectified Linear Unit (ReLU) activation function element-wise.

        Returns:
            Tensor: New Tensor instance with ReLU applied element-wise.
        """
        return Tensor(np.maximum(self.data, 0))

    def sigmoid(self):
        """
        Apply the sigmoid activation function element-wise.

        Returns:
            Tensor: New Tensor instance with sigmoid applied element-wise.
        """
        return Tensor(1 / (1 + np.exp(-self.data)))

    def tanh(self):
        """
        Apply the hyperbolic tangent (tanh) activation function element-wise.

        Returns:
            Tensor: New Tensor instance with tanh applied element-wise.
        """
        return Tensor(np.tanh(self.data))
    
    def exp(self):
        """
        Compute the element-wise exponentiation of the tensor.

        Returns:
            Tensor: New Tensor instance with exponentiation applied element-wise.
        """
        return Tensor(np.exp(self.data))

    def log(self):
        """
        Compute the element-wise natural logarithm of the tensor.

        Returns:
            Tensor: New Tensor instance with logarithm applied element-wise.
        Note:
            Be cautious when using this function with non-positive values.
            np.finfo(self.dtype).tiny is used to retrieve the smallest positive normalized number representable by the tensor's data type.
            This value is used to replace any non-positive values before applying the logarithm operation.
        """
        if np.any(self.data <= 0):
            print("Warning: Logarithm operation encountered non-positive values. Results may be invalid.")
        return Tensor(np.log(np.maximum(self.data, np.finfo(self.dtype).tiny)))
    
    def __eq__(self, other):
        """
        Perform element-wise equality comparison with another tensor or value.

        Args:
            other (Tensor or value): Another tensor or value to compare element-wise.

        Returns:
            Tensor: New Tensor instance containing boolean values indicating equality.
        """
        if isinstance(other, Tensor):
            return Tensor(self.data == other.data)
        else:
            return Tensor(self.data == other)

    def __lt__(self, other):
        """
        Perform element-wise less-than comparison with another tensor or value.

        Args:
            other (Tensor or value): Another tensor or value to compare element-wise.

        Returns:
            Tensor: New Tensor instance containing boolean values indicating less-than comparison.
        """
        if isinstance(other, Tensor):
            return Tensor(self.data < other.data)
        else:
            return Tensor(self.data < other)

    def __le__(self, other):
        """
        Perform element-wise less-than or equal-to comparison with another tensor or value.

        Args:
            other (Tensor or value): Another tensor or value to compare element-wise.

        Returns:
            Tensor: New Tensor instance containing boolean values indicating less-than or equal-to comparison.
        """
        if isinstance(other, Tensor):
            return Tensor(self.data <= other.data)
        else:
            return Tensor(self.data <= other)

    def __gt__(self, other):
        """
        Perform element-wise greater-than comparison with another tensor or value.

        Args:
            other (Tensor or value): Another tensor or value to compare element-wise.

        Returns:
            Tensor: New Tensor instance containing boolean values indicating greater-than comparison.
        """
        if isinstance(other, Tensor):
            return Tensor(self.data > other.data)
        else:
            return Tensor(self.data > other)

    def __ge__(self, other):
        """
        Perform element-wise greater-than or equal-to comparison with another tensor or value.

        Args:
            other (Tensor or value): Another tensor or value to compare element-wise.

        Returns:
            Tensor: New Tensor instance containing boolean values indicating greater-than or equal-to comparison.
        """
        if isinstance(other, Tensor):
            return Tensor(self.data >= other.data)
        else:
            return Tensor(self.data >= other)

    def __str__(self):
        """
        Get a string representation of the tensor.

        Returns:
            str: A human-readable string representing the tensor's data, shape, and data type.
        """
        return f"Tensor(data={self.data}, shape={self.shape}, dtype={self.dtype})"
