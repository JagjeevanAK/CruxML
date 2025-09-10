"""
Example: Basic Usage of CruxML

This script demonstrates the basic usage of the CruxML library
to create and train a simple neural network.
"""

from cruxml import Value, MLP, Trainer, MSELoss


def main():
    print("CruxML Basic Example")
    print("=" * 40)
    
    # Create training data (XOR problem)
    X_train = [
        [2.0, 3.0, -1.0],
        [3.0, -1.0, 0.5],
        [0.5, 1.0, 1.0],
        [1.0, 1.0, -1.0]
    ]
    y_train = [1.0, -1.0, -1.0, 1.0]
    
    print("Training Data:")
    for i, (x, y) in enumerate(zip(X_train, y_train)):
        print(f"  Sample {i+1}: {x} -> {y}")
    print()
    
    # Create a neural network
    # 3 inputs -> 4 hidden -> 4 hidden -> 1 output
    model = MLP(3, [4, 4, 1])
    print(f"Created MLP with {len(model.parameters())} parameters")
    
    # Create trainer
    trainer = Trainer(model, MSELoss(), learning_rate=0.1)
    
    # Train the model
    print("\nTraining...")
    history = trainer.train(X_train, y_train, epochs=100, verbose=True)
    
    # Make predictions
    print("\nFinal Predictions:")
    predictions = trainer.predict(X_train)
    for i, (x, y_true, y_pred) in enumerate(zip(X_train, y_train, predictions)):
        pred_val = y_pred[0].data if isinstance(y_pred, list) else y_pred.data
        print(f"  Sample {i+1}: Input={x}, True={y_true}, Predicted={pred_val:.4f}")
    
    # Evaluate final performance
    final_loss = trainer.evaluate(X_train, y_train)
    print(f"\nFinal Training Loss: {final_loss:.6f}")


if __name__ == "__main__":
    main()
