"""
CruxML Autograd Engine

A lightweight automatic differentiation engine for neural networks.
"""

import math


class Value:
    """
    A scalar value with automatic differentiation capabilities.
    
    This class wraps a scalar value and tracks operations for automatic
    gradient computation via backpropagation.
    """
    
    def __init__(self, data, _children=(), _op='', label=''):
        """
        Initialize a Value object.
        
        Args:
            data: The scalar value
            _children: Tuple of parent Value objects
            _op: String representing the operation that created this value
            label: Optional label for debugging/visualization
        """
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __repr__(self) -> str:
        return f"Value(data={self.data})"

    def __add__(self, other):
        """Addition operation with automatic differentiation."""
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward

        return out

    def __mul__(self, other):
        """Multiplication operation with automatic differentiation."""
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward

        return out

    def __rmul__(self, other):
        """Reverse multiplication (for scalar * Value)."""
        return self * other

    def __truediv__(self, other):
        """Division operation."""
        return self * other**-1

    def __pow__(self, other):
        """Power operation with automatic differentiation."""
        assert isinstance(other, (int, float)), "Only support int/float powers"
        out = Value(self.data**other, (self,), f'**{other}')

        def _backward():
            self.grad += other * (self.data ** (other - 1)) * out.grad
        out._backward = _backward

        return out

    def __neg__(self):
        """Negation operation."""
        return self * -1

    def __sub__(self, other):
        """Subtraction operation."""
        other = other if isinstance(other, Value) else Value(other)
        return self + (-other)

    def tanh(self):
        """Hyperbolic tangent activation function."""
        x = self.data
        t = (math.exp(2*x) - 1) / (math.exp(2*x) + 1)
        out = Value(t, (self,), 'tanh')

        def _backward():
            self.grad += (1 - t**2) * out.grad
        out._backward = _backward

        return out

    def exp(self):
        """Exponential function."""
        x = self.data
        out = Value(math.exp(x), (self,), 'exp')

        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward

        return out

    def relu(self):
        """ReLU activation function."""
        out = Value(0 if self.data < 0 else self.data, (self,), 'ReLU')

        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward

        return out

    def sigmoid(self):
        """Sigmoid activation function."""
        x = self.data
        s = 1 / (1 + math.exp(-x))
        out = Value(s, (self,), 'sigmoid')

        def _backward():
            self.grad += s * (1 - s) * out.grad
        out._backward = _backward

        return out

    def log(self):
        """Natural logarithm function."""
        x = self.data
        out = Value(math.log(x), (self,), 'log')

        def _backward():
            self.grad += (1.0 / self.data) * out.grad
        out._backward = _backward

        return out

    def backward(self):
        """
        Compute gradients via backpropagation.
        
        Performs topological sort and computes gradients for all
        nodes in the computational graph.
        """
        topo = []
        visited = set()
        
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        
        build_topo(self)

        self.grad = 1.0
        for node in reversed(topo):
            node._backward()
