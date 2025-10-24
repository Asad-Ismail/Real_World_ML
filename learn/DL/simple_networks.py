
import numpy as np

# =============================================================================
# PATTERN: All models follow these steps
# =============================================================================
# 1. Forward pass: compute predictions
# 2. Compute loss
# 3. Backward pass:
#    a) Compute error at output: dL/d(output)
#    b) Propagate error backwards through each layer
#    c) Compute weight gradients: activation^T @ error
#    d) Update weights
# =============================================================================


class LinearRegression:
    """Simplest case: y = Wx + b"""
    
    def __init__(self, input_dim):
        self.W = np.random.randn(input_dim, 1) * 0.01
        self.b = np.zeros((1, 1))
    
    def forward(self, X):
        self.X = X
        return X @ self.W + self.b
    
    def backward(self, y_pred, y_true, lr=0.01):
        n = len(y_pred)
        
        # Step 1: Compute error at output
        error = (y_pred - y_true) / n
        
        # Step 2: Compute gradients (error × activation)
        dW = self.X.T @ error  # (input_dim, n) @ (n, 1) = (input_dim, 1)
        db = np.sum(error, axis=0, keepdims=True)
        
        # Step 3: Update
        self.W -= lr * dW
        self.b -= lr * db


class LogisticRegression:
    """Add sigmoid activation: y = sigmoid(Wx + b)"""
    
    def __init__(self, input_dim):
        self.W = np.random.randn(input_dim, 1) * 0.01
        self.b = np.zeros((1, 1))
    
    def forward(self, X):
        self.X = X
        z = X @ self.W + self.b
        self.y = 1 / (1 + np.exp(-z))  # sigmoid
        return self.y
    
    def backward(self, y_pred, y_true, lr=0.01):
        n = len(y_pred)
        
        # Step 1: Compute error at output
        # For sigmoid + cross-entropy: dL/dz = y - t (simplified!)
        error = (y_pred - y_true) / n
        
        # Step 2: Compute gradients (same pattern!)
        dW = self.X.T @ error
        db = np.sum(error, axis=0, keepdims=True)
        
        # Step 3: Update
        self.W -= lr * dW
        self.b -= lr * db


class MLP:
    """Add hidden layer: y = W2 @ sigmoid(W1 @ x + b1) + b2"""
    
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.W1 = np.random.randn(input_dim, hidden_dim) * 0.01
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = np.random.randn(hidden_dim, output_dim) * 0.01
        self.b2 = np.zeros((1, output_dim))
    
    def forward(self, X):
        self.X = X
        # Layer 1
        z1 = X @ self.W1 + self.b1
        self.h = 1 / (1 + np.exp(-z1))  # sigmoid
        # Layer 2
        return self.h @ self.W2 + self.b2
    
    def backward(self, y_pred, y_true, lr=0.01):
        n = len(y_pred)
        
        # Step 1: Compute error at output layer
        error_output = (y_pred - y_true) / n
        
        # Step 2a: Gradients for output layer (same pattern!)
        dW2 = self.h.T @ error_output
        db2 = np.sum(error_output, axis=0, keepdims=True)
        
        # Step 2b: Backpropagate error to hidden layer
        error_hidden = error_output @ self.W2.T
        
        # Step 2c: Apply activation derivative
        error_hidden = error_hidden * self.h * (1 - self.h)
        
        # Step 2d: Gradients for hidden layer (same pattern!)
        dW1 = self.X.T @ error_hidden
        db1 = np.sum(error_hidden, axis=0, keepdims=True)
        
        # Step 3: Update all weights
        self.W2 -= lr * dW2
        self.b2 -= lr * db2
        self.W1 -= lr * dW1
        self.b1 -= lr * db1


# =============================================================================
# THE PATTERN (Generic Algorithm)
# =============================================================================
"""
FOR EACH LAYER (from output to input):
    
    1. Receive error from next layer (or compute from loss)
       error_current = error_from_next_layer
    
    2. Compute weight gradient:
       dW = (activation_from_previous_layer)^T @ error_current
       db = sum(error_current)
    
    3. If not the first layer:
       Propagate error backwards:
       error_previous = error_current @ W_current^T
       
       If previous layer has activation:
       error_previous = error_previous * activation_derivative
    
    4. Update weights:
       W -= learning_rate * dW
       b -= learning_rate * db

REPEAT for all layers!
"""

# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("BACKPROPAGATION PATTERN DEMONSTRATION")
    print("=" * 70)
    
    # Generate toy data
    np.random.seed(42)
    X = np.random.randn(100, 3)  # 100 samples, 3 features
    y = np.random.randn(100, 1)  # 100 targets
    
    print("\n1. LINEAR REGRESSION")
    print("-" * 70)
    model1 = LinearRegression(input_dim=3)
    for i in range(3):
        y_pred = model1.forward(X)
        loss = np.mean((y_pred - y) ** 2) / 2
        print(f"   Iteration {i}: Loss = {loss:.4f}")
        model1.backward(y_pred, y, lr=0.01)
    
    print("\n2. LOGISTIC REGRESSION")
    print("-" * 70)
    y_binary = (y > 0).astype(float)  # Binary labels
    model2 = LogisticRegression(input_dim=3)
    for i in range(3):
        y_pred = model2.forward(X)
        # Binary cross-entropy
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        loss = -np.mean(y_binary * np.log(y_pred) + 
                       (1 - y_binary) * np.log(1 - y_pred))
        print(f"   Iteration {i}: Loss = {loss:.4f}")
        model2.backward(y_pred, y_binary, lr=0.1)
    
    print("\n3. MULTI-LAYER PERCEPTRON")
    print("-" * 70)
    model3 = MLP(input_dim=3, hidden_dim=5, output_dim=1)
    for i in range(3):
        y_pred = model3.forward(X)
        loss = np.mean((y_pred - y) ** 2) / 2
        print(f"   Iteration {i}: Loss = {loss:.4f}")
        model3.backward(y_pred, y, lr=0.01)
    
    print("\n" + "=" * 70)
    print("KEY INSIGHT: Same pattern, just more layers!")
    print("=" * 70)
    print("""
    The fundamental algorithm is identical:
    
    1. Compute output error
    2. For each layer (back to front):
       - Compute weight gradient: activation^T @ error
       - Backpropagate error: error @ W^T
       - Apply activation derivative if needed
    3. Update all weights
    
    This pattern scales to ANY number of layers!
    """)