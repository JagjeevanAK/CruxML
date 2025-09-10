# CruxML: Lightweight Deep Learning Framework

A minimal PyTorch-like deep learning library built from scratch for educational purposes and rapid prototyping. CruxML provides a clean, readable implementation of core neural network concepts including automatic differentiation, neural network modules, and training utilities.

## Overview

This implementation provides a complete deep learning framework featuring:

• **Automatic Differentiation Engine**: Scalar-based autograd system with full backpropagation support
• **Neural Network Components**: Modular neuron, layer, and multi-layer perceptron implementations
• **Loss Functions**: Mean Squared Error, Mean Absolute Error, and Binary Cross Entropy
• **Training Infrastructure**: Complete trainer with progress tracking and evaluation
• **Visualization Tools**: Computational graph visualization and training progress plots
• **Educational Design**: Clean, well-documented codebase perfect for learning neural network internals

## Architecture Components

### Core Modules

• `autograd.py`: Automatic differentiation engine with Value class
• `nn.py`: Neural network building blocks (Neuron, Layer, MLP)
• `losses.py`: Loss function implementations
• `trainer.py`: Training loops and model evaluation
• `visualization.py`: Graph visualization and plotting utilities

### Key Features

#### Automatic Differentiation

The core autograd engine implements scalar automatic differentiation:

```python
from cruxml import Value

# Create values with gradient tracking
a = Value(2.0, label='a')
b = Value(-3.0, label='b')
c = a * b + Value(10.0)

# Compute gradients via backpropagation
c.backward()
print(f"∂c/∂a = {a.grad}")  # ∂c/∂a = -3.0
print(f"∂c/∂b = {b.grad}")  # ∂c/∂b = 2.0
```

Supports all essential operations:
• Arithmetic operations: addition, multiplication, division, power
• Activation functions: tanh, ReLU, sigmoid, exponential, logarithm
• Proper gradient computation through computational graph

#### Neural Network Architecture

Multi-layer perceptron with configurable architecture:

```python
from cruxml import MLP

# Create a 3-layer network: 3 inputs → 4 hidden → 4 hidden → 1 output
model = MLP(3, [4, 4, 1], activations=['tanh', 'tanh', 'tanh'])

# Forward pass
output = model([1.0, 2.0, -1.0])
```

Features:
• Flexible layer configuration with custom activation functions
• Parameter management and gradient computation
• Modular design with separate Neuron and Layer classes

#### Training System

Complete training infrastructure with progress tracking:

```python
from cruxml import Trainer, MSELoss

# Setup training
trainer = Trainer(model, MSELoss(), learning_rate=0.1)

# Train the model
history = trainer.train(X_train, y_train, epochs=100)

# Evaluate and predict
test_loss = trainer.evaluate(X_test, y_test)
predictions = trainer.predict(X_new)
```

## Installation

This project requires Python 3.12+ and uses modern dependency management.

```bash
# Clone the repository
git clone https://github.com/JagjeevanAK/CruxML.git
cd CruxML

# Install in development mode
pip install -e .
```

### Dependencies

• **NumPy**: Numerical computing foundation
• **Matplotlib**: Plotting and visualization (optional)
• **Graphviz**: Computational graph visualization (optional)

## Usage

### Basic Example

```python
from cruxml import Value, MLP, Trainer, MSELoss

# Create training data (XOR problem)
X_train = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0]
]
y_train = [1.0, -1.0, -1.0, 1.0]

# Build neural network
model = MLP(3, [4, 4, 1])  # 3 inputs, two hidden layers of 4 neurons, 1 output

# Train the model
trainer = Trainer(model, MSELoss(), learning_rate=0.1)
history = trainer.train(X_train, y_train, epochs=100)

# Make predictions
predictions = trainer.predict(X_train)
```

### Advanced Usage

```python
# Custom activation functions
model = MLP(3, [8, 4, 1], activations=['relu', 'tanh', 'sigmoid'])

# Different loss functions
from cruxml import MAELoss, BCELoss
trainer = Trainer(model, MAELoss(), learning_rate=0.05)

# Visualization
from cruxml import draw_graph, plot_training_history
draw_graph(loss_value, filename='computational_graph')
plot_training_history(history)
```

## Project Structure

```
src/cruxml/
├── __init__.py             # Main package interface
├── autograd.py             # Automatic differentiation engine
├── nn.py                   # Neural network components
├── losses.py               # Loss function implementations
├── trainer.py              # Training and evaluation utilities
├── visualization.py        # Plotting and graph visualization
└── py.typed               # Type checking support

examples/
├── basic_example.py        # XOR problem demonstration
└── autograd_demo.py        # Autograd engine examples
```

## Implementation Details

### Computational Graph

The autograd engine builds a computational graph during forward pass and computes gradients via topological sorting:

• Tracks operations and dependencies between Value objects
• Implements proper chain rule for gradient computation
• Supports arbitrary computational graph structures

### Training Considerations

• Uses gradient descent optimization with configurable learning rates
• Implements proper parameter updates and gradient zeroing
• Supports batch training and evaluation metrics

### Performance

• Optimized for educational clarity over raw performance
• Scalar-based operations for transparency
• Memory-efficient gradient computation

## Technical Specifications

| Component | Implementation |
|-----------|----------------|
| Differentiation | Scalar automatic differentiation |
| Optimization | Gradient descent |
| Activations | tanh, ReLU, sigmoid, linear |
| Loss Functions | MSE, MAE, BCE |
| Architecture | Feedforward neural networks |
| Backend | Pure Python with NumPy |

## Examples

The `examples/` directory contains practical demonstrations:

• **basic_example.py**: Complete XOR problem solution
• **autograd_demo.py**: Autograd engine functionality showcase

Run examples with:
```bash
cd examples
PYTHONPATH=../src python basic_example.py
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is open source and available under the MIT License.

## References

• Karpathy, A. "micrograd" - Inspiration for scalar autograd implementation
• Goodfellow, I., Bengio, Y., & Courville, A. (2016). "Deep Learning"
• PyTorch Documentation - API design reference

This implementation is for educational and research purposes, providing a clear understanding of neural network fundamentals and automatic differentiation.
