"""
Loss Functions for CruxML

Common loss functions for training neural networks.
"""

from .autograd import Value


class Loss:
    """Base class for loss functions."""
    
    def __call__(self, predictions, targets):
        """
        Compute the loss.
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            
        Returns:
            Loss value
        """
        raise NotImplementedError


class MSELoss(Loss):
    """Mean Squared Error Loss."""
    
    def __call__(self, predictions, targets):
        """
        Compute MSE loss.
        
        Args:
            predictions: List of predicted values
            targets: List of target values
            
        Returns:
            MSE loss as a Value object
        """
        if not isinstance(predictions, list):
            predictions = [predictions]
        if not isinstance(targets, list):
            targets = [targets]
            
        losses = []
        for pred, target in zip(predictions, targets):
            if isinstance(pred, list) and len(pred) == 1:
                pred = pred[0]
            if not isinstance(pred, Value):
                pred = Value(pred)
            target_val = target if isinstance(target, Value) else Value(target)
            loss = (pred - target_val) ** 2
            losses.append(loss)
        
        # Sum all losses
        total_loss = Value(0.0)
        for loss in losses:
            total_loss = total_loss + loss
            
        return total_loss


class MAELoss(Loss):
    """Mean Absolute Error Loss."""
    
    def __call__(self, predictions, targets):
        """
        Compute MAE loss.
        
        Args:
            predictions: List of predicted values
            targets: List of target values
            
        Returns:
            MAE loss as a Value object
        """
        if not isinstance(predictions, list):
            predictions = [predictions]
        if not isinstance(targets, list):
            targets = [targets]
            
        losses = []
        for pred, target in zip(predictions, targets):
            if isinstance(pred, list) and len(pred) == 1:
                pred = pred[0]
            if not isinstance(pred, Value):
                pred = Value(pred)
            target_val = target if isinstance(target, Value) else Value(target)
            diff = pred - target_val
            # Approximate absolute value using smooth approximation
            # |x| ≈ sqrt(x^2 + ε) where ε is small
            epsilon = Value(1e-8)
            abs_diff = (diff ** 2 + epsilon) ** 0.5
            losses.append(abs_diff)
        
        # Sum all losses
        total_loss = Value(0.0)
        for loss in losses:
            total_loss = total_loss + loss
            
        return total_loss


class BCELoss(Loss):
    """Binary Cross Entropy Loss (for binary classification)."""
    
    def __call__(self, predictions, targets):
        """
        Compute simplified BCE loss using MSE approximation.
        
        Args:
            predictions: List of predicted probabilities (should be sigmoid outputs)
            targets: List of binary targets (0 or 1)
            
        Returns:
            BCE loss as a Value object
        """
        # For simplicity, we'll use MSE loss for now
        # In practice, you'd want proper BCE implementation
        mse = MSELoss()
        return mse(predictions, targets)
