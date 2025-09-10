"""
Example: Autograd Engine Demo

This script demonstrates the automatic differentiation capabilities
of the CruxML Value class.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from cruxml import Value


def main():
    print("CruxML Autograd Demo")
    print("=" * 40)
    
    # Create some values
    a = Value(2.0, label='a')
    b = Value(-3.0, label='b')
    c = Value(10.0, label='c')
    
    print("Initial values:")
    print(f"  a = {a}")
    print(f"  b = {b}")
    print(f"  c = {c}")
    print()
    
    # Perform some operations
    d = a * b
    d.label = 'd'
    e = d + c
    e.label = 'e'
    f = Value(-2.0, label='f')
    l = e * f
    l.label = 'l'
    
    print("Operations:")
    print(f"  d = a * b = {d}")
    print(f"  e = d + c = {e}")
    print(f"  f = {f}")
    print(f"  l = e * f = {l}")
    print()
    
    # Compute gradients
    print("Computing gradients...")
    l.backward()
    
    print("Gradients:")
    print(f"  a.grad = {a.grad}")
    print(f"  b.grad = {b.grad}")
    print(f"  c.grad = {c.grad}")
    print(f"  d.grad = {d.grad}")
    print(f"  e.grad = {e.grad}")
    print(f"  f.grad = {f.grad}")
    print(f"  l.grad = {l.grad}")
    print()
    
    # Demonstrate gradient descent step
    print("Performing gradient descent step (learning rate = 0.01):")
    lr = 0.01
    a.data += -lr * a.grad
    b.data += -lr * b.grad
    c.data += -lr * c.grad
    f.data += -lr * f.grad
    
    # Recompute with updated values
    d_new = a * b
    e_new = d_new + c
    l_new = e_new * f
    
    print(f"  New l value: {l_new.data} (was {l.data})")
    
    # Demonstrate activation functions
    print("\nActivation Functions Demo:")
    x = Value(0.5, label='x')
    print(f"  x = {x}")
    print(f"  tanh(x) = {x.tanh()}")
    print(f"  sigmoid(x) = {x.sigmoid()}")
    print(f"  relu(x) = {x.relu()}")
    print(f"  exp(x) = {x.exp()}")


if __name__ == "__main__":
    main()
