"""
Neural Network Components for CruxML

This module contains the core neural network building blocks:
- Neuron: Individual neuron with weights and bias
- Layer: Collection of neurons forming a layer
- MLP: Multi-layer perceptron (feedforward neural network)
"""

import random
from .autograd import Value


class Neuron:
    """
    A single neuron with learnable weights and bias.
    
    Implements the basic neuron computation: activation(sum(w_i * x_i) + bias)
    """
    
    def __init__(self, nin, activation='tanh'):
        """
        Initialize a neuron.
        
        Args:
            nin: Number of input connections
            activation: Activation function ('tanh', 'relu', 'sigmoid')
        """
        self.weights = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.bias = Value(random.uniform(-1, 1))
        self.activation = activation

    def __call__(self, x):
        """
        Forward pass through the neuron.
        
        Args:
            x: List of input values
            
        Returns:
            Activated output value
        """
        # Weighted sum of inputs plus bias
        activation_input = sum((wi * xi for wi, xi in zip(self.weights, x)), self.bias)
        
        # Apply activation function
        if self.activation == 'tanh':
            return activation_input.tanh()
        elif self.activation == 'relu':
            return activation_input.relu()
        elif self.activation == 'sigmoid':
            return activation_input.sigmoid()
        else:
            return activation_input  # Linear activation

    def parameters(self):
        """Return all learnable parameters."""
        return self.weights + [self.bias]


class Layer:
    """
    A layer of neurons.
    
    Groups multiple neurons together to form a layer in the neural network.
    """
    
    def __init__(self, nin, nout, activation='tanh'):
        """
        Initialize a layer.
        
        Args:
            nin: Number of input features
            nout: Number of output neurons
            activation: Activation function for all neurons in this layer
        """
        self.neurons = [Neuron(nin, activation) for _ in range(nout)]

    def __call__(self, x):
        """
        Forward pass through the layer.
        
        Args:
            x: List of input values
            
        Returns:
            List of outputs from each neuron, or single value if only one neuron
        """
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        """Return all learnable parameters in this layer."""
        params = []
        for neuron in self.neurons:
            params.extend(neuron.parameters())
        return params


class MLP:
    """
    Multi-Layer Perceptron (Feedforward Neural Network).
    
    A sequence of fully connected layers that can learn complex patterns.
    """
    
    def __init__(self, nin, nouts, activations=None):
        """
        Initialize the MLP.
        
        Args:
            nin: Number of input features
            nouts: List of output sizes for each layer
            activations: List of activation functions for each layer
                        If None, uses 'tanh' for all layers
        """
        if activations is None:
            activations = ['tanh'] * len(nouts)
        elif len(activations) != len(nouts):
            raise ValueError("Number of activations must match number of layers")
            
        sz = [nin] + nouts
        self.layers = [
            Layer(sz[i], sz[i+1], activations[i]) 
            for i in range(len(nouts))
        ]

    def __call__(self, x):
        """
        Forward pass through the entire network.
        
        Args:
            x: List of input values
            
        Returns:
            Network output
        """
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        """Return all learnable parameters in the network."""
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params

    def zero_grad(self):
        """Zero out all gradients."""
        for p in self.parameters():
            p.grad = 0.0

    def update_parameters(self, learning_rate):
        """
        Update parameters using gradient descent.
        
        Args:
            learning_rate: Learning rate for parameter updates
        """
        for p in self.parameters():
            p.data += -learning_rate * p.grad
