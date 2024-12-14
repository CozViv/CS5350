from NeuralNetwork import setNeuralNetwork
import numpy as np

X = np.array([[1],[1],[1]])
y = np.array([[1]])

nn = setNeuralNetwork()

cache = nn.forward(X)
gradients = nn.backpropagate(X, y, cache)

print("Gradients:")
print("dW0:\n", gradients['W0'])
print("db0:\n", gradients['b0'])
print("dW1:\n", gradients['W1'])
print("db1:\n", gradients['b1'])
print("dW2:\n", gradients['W2'])
print("db2:\n", gradients['b2'])
