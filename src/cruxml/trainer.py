"""
Training utilities for CruxML

This module provides training loops and utilities for neural networks.
"""

from .losses import MSELoss


class Trainer:
    """
    A trainer class to handle the training process for neural networks.
    """
    
    def __init__(self, model, loss_fn=None, learning_rate=0.01):
        """
        Initialize the trainer.
        
        Args:
            model: The neural network model to train
            loss_fn: Loss function to use (defaults to MSELoss)
            learning_rate: Learning rate for parameter updates
        """
        self.model = model
        self.loss_fn = loss_fn or MSELoss()
        self.learning_rate = learning_rate
        self.history = {'loss': []}

    def train_step(self, X_batch, y_batch):
        """
        Perform a single training step.
        
        Args:
            X_batch: Batch of input data
            y_batch: Batch of target values
            
        Returns:
            Loss value for this batch
        """
        # Forward pass
        predictions = [self.model(x) for x in X_batch]
        
        # Compute loss
        loss = self.loss_fn(predictions, y_batch)
        
        # Zero gradients
        self.model.zero_grad()
        
        # Backward pass
        loss.backward()
        
        # Update parameters
        self.model.update_parameters(self.learning_rate)
        
        return loss

    def train(self, X_train, y_train, epochs=100, verbose=True):
        """
        Train the model for multiple epochs.
        
        Args:
            X_train: Training input data
            y_train: Training target values
            epochs: Number of training epochs
            verbose: Whether to print training progress
            
        Returns:
            Training history (loss values)
        """
        for epoch in range(epochs):
            loss = self.train_step(X_train, y_train)
            self.history['loss'].append(loss.data)
            
            if verbose and (epoch % (epochs // 10) == 0 or epoch == epochs - 1):
                print(f"Epoch {epoch:4d}/{epochs}: Loss = {loss.data:.6f}")
        
        return self.history

    def evaluate(self, X_test, y_test):
        """
        Evaluate the model on test data.
        
        Args:
            X_test: Test input data
            y_test: Test target values
            
        Returns:
            Test loss value
        """
        predictions = [self.model(x) for x in X_test]
        loss = self.loss_fn(predictions, y_test)
        return loss.data

    def predict(self, X):
        """
        Make predictions on new data.
        
        Args:
            X: Input data
            
        Returns:
            Model predictions
        """
        if not isinstance(X[0], list):
            # Single sample
            return self.model(X)
        else:
            # Multiple samples
            return [self.model(x) for x in X]
