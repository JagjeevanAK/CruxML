"""
CruxML - A lightweight neural network library with automatic differentiation

This library provides:
- Automatic differentiation engine (Value class)
- Neural network components (Neuron, Layer, MLP)
- Loss functions (MSELoss, MAELoss, BCELoss)
- Training utilities (Trainer)
- Visualization tools

Example usage:
    from cruxml import Value, MLP, Trainer, MSELoss
    
    # Create a simple neural network
    model = MLP(3, [4, 4, 1])  # 3 inputs, two hidden layers of 4 neurons, 1 output
    
    # Prepare training data
    X_train = [[2.0, 3.0, -1.0], [3.0, -1.0, 0.5], [0.5, 1.0, 1.0], [1.0, 1.0, -1.0]]
    y_train = [1.0, -1.0, -1.0, 1.0]
    
    # Train the model
    trainer = Trainer(model, MSELoss(), learning_rate=0.1)
    history = trainer.train(X_train, y_train, epochs=100)
    
    # Make predictions
    predictions = trainer.predict(X_train)
"""

# Core autograd functionality
from .autograd import Value

# Neural network components
from .nn import Neuron, Layer, MLP

# Loss functions
from .losses import Loss, MSELoss, MAELoss, BCELoss

# Training utilities
from .trainer import Trainer

# Visualization (optional - depends on external libraries)
try:
    from .visualization import draw_graph, plot_training_history, plot_predictions
except ImportError:
    # Visualization not available if dependencies are missing
    pass

def hello() -> str:
    return "Hello from cruxml!"

__version__ = "0.1.0"
__author__ = "Your Name"

__all__ = [
    # Core
    "Value",
    # Neural Networks
    "Neuron", "Layer", "MLP",
    # Losses
    "Loss", "MSELoss", "MAELoss", "BCELoss",
    # Training
    "Trainer",
    # Utilities
    "hello",
    # Visualization (if available)
    "draw_graph", "plot_training_history", "plot_predictions"
]
