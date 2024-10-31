import numpy as np

# OR operation data
# Inputs and expected outputs for the OR gate
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 1])

# Initialize weights and bias randomly within the range (-0.5, 0.5)
weights = np.random.uniform(-0.5, 0.5, 2)
bias = np.random.uniform(-0.5, 0.5)
learning_rate = 0.1
threshold = 0.2
epochs = 10

# Activation function (includes threshold)
def step_function(x, theta):
    return 1 if x >= theta else 0

# Training the perceptron
for epoch in range(epochs):
    print(f"Epoch {epoch + 1}")
    for i in range(len(X)):
        # Calculate linear combination of inputs and weights
        linear_combination = np.dot(X[i], weights) + bias
        # Apply activation function with threshold
        y_pred = step_function(linear_combination, threshold)
        # Calculate the error
        error = y[i] - y_pred
        # Update weights and bias
        weights += learning_rate * error * X[i]
        bias += learning_rate * error

        # Print the updated weights and bias after each example
        print(f"Input: {X[i]}, Expected: {y[i]}, Predicted: {y_pred}, Weights: {weights}, Bias: {bias}")
    
    print("\n")

# Test the final weights and bias
print("Final weights:", weights)
print("Final bias:", bias)

# Testing the perceptron after training
print("\nTesting trained perceptron:")
for i in range(len(X)):
    linear_combination = np.dot(X[i], weights) + bias
    y_pred = step_function(linear_combination, threshold)
    print(f"Input: {X[i]}, Predicted Output: {y_pred}")
