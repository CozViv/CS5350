import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, hidden_layer_size, random=True):
        self.hidden_layer_size = hidden_layer_size
        if random:
            self.W0 = np.random.randn(hidden_layer_size, input_size)
            self.b0 = np.random.randn(hidden_layer_size, 1)
            self.W1 = np.random.randn(hidden_layer_size, hidden_layer_size)
            self.b1 = np.random.randn(hidden_layer_size, 1)
            self.W2 = np.random.randn(hidden_layer_size, hidden_layer_size)
            self.b2 = np.random.randn(hidden_layer_size, 1)
            self.W3 = np.random.randn(1, hidden_layer_size)
            self.b3 = np.random.randn(1, 1)
        else:
            self.W0 = np.zeros((hidden_layer_size, input_size))
            self.b0 = np.zeros((hidden_layer_size, 1))
            self.W1 = np.zeros((hidden_layer_size, hidden_layer_size))
            self.b1 = np.zeros((hidden_layer_size, 1))
            self.W2 = np.zeros((hidden_layer_size, hidden_layer_size))
            self.b2 = np.zeros((hidden_layer_size, 1))
            self.W3 = np.zeros((1, hidden_layer_size))
            self.b3 = np.zeros((1, 1))
        
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -100, 100)))

    def sigmoid_derivative(self, x):
        sig = self.sigmoid(x)
        return sig * (1 - sig)
    
    def forward(self, X):
        z0 = self.W0 @ X + self.b0
        a0 = self.sigmoid(z0)
        z1 = self.W1 @ a0 + self.b1
        a1 = self.sigmoid(z1)
        z2 = self.W2 @ a1 + self.b2
        a2 = self.sigmoid(z2)
        z3 = self.W3 @ a2 + self.b3
        a3 = z3
        return {'z0': z0, 'a0': a0, 'z1': z1, 'a1': a1, 'z2': z2, 'a2': a2, 'z3': z3, 'a3': a3}
    
    def back_propagation(self, X, y, cache):
        x_shape = X.shape[1]
        
        dz3 = cache['a3'] - y
        dW3 = dz3 @ cache['a2'].T / x_shape
        db3 = np.sum(dz3, axis=1, keepdims=True) / x_shape
        
        dz2 = (self.W3.T @ dz3) * self.sigmoid_derivative(cache['z2'])
        dW2 = dz2 @ cache['a1'].T / x_shape
        db2 = np.sum(dz2, axis=1, keepdims=True) / x_shape
        
        dz1 = (self.W2.T @ dz2) * self.sigmoid_derivative(cache['z1'])
        dW1 = dz1 @ cache['a0'].T / x_shape
        db1 = np.sum(dz1, axis=1, keepdims=True) / x_shape
        
        dz0 = (self.W1.T @ dz1) * self.sigmoid_derivative(cache['z0'])
        dW0 = dz0 @ X.T / x_shape
        db0 = np.sum(dz0, axis=1, keepdims=True) / x_shape
        
        return {'W3': dW3, 'b3': db3, 'W2': dW2, 'b2': db2, 'W1': dW1, 'b1': db1, 'W0': dW0, 'b0': db0}
    
    def update_weights(self, grads, learning_rate):
        self.W0 -= learning_rate * grads['W0']
        self.b0 -= learning_rate * grads['b0']
        self.W1 -= learning_rate * grads['W1']
        self.b1 -= learning_rate * grads['b1']
        self.W2 -= learning_rate * grads['W2']
        self.b2 -= learning_rate * grads['b2']
        self.W3 -= learning_rate * grads['W3']
        self.b3 -= learning_rate * grads['b3']
    
    def train(self, X_train, y_train, epochs, gamma0, d):
        losses = []
        for epoch in range(epochs):
            size_samples = len(X_train)
            rand_indices = np.random.permutation(size_samples)
            X_train = X_train.iloc[rand_indices].reset_index(drop=True)
            y_train = y_train.iloc[rand_indices].reset_index(drop=True)

            for i in range(len(X_train)):
                lr = gamma0 / (1 + gamma0 * epoch / d)
                X = X_train.iloc[i].values.reshape(-1, 1)
                y = np.array([[y_train.iloc[i]]])

                cache = self.forward(X)
                grads = self.back_propagation(X, y, cache)
                self.update_weights(grads, lr)

            y_pred = []
            for i in range(len(X_train)):
                X_i = X_train.iloc[i].values.reshape(-1, 1)
                output = self.forward(X_i)
                y_pred.append(output['a3'].item())
            y_pred = np.array(y_pred).reshape(-1, 1)
            y_true = y_train.values.reshape(-1, 1)
            loss = np.mean((y_true - y_pred) ** 2)
            losses.append(loss)
        return losses
    
    def predict_all(self, X):
        predictions = []
        for i in range(len(X)):
            X_i = X.iloc[i].values.reshape(-1, 1)
            output = self.forward(X_i)
            prediction = 1 if output['a3'] > .5 else 0
            predictions.append(prediction)
        return predictions

class setNeuralNetwork:
    def __init__(self):
        self.W0 = np.array([[-1, 1], [-2, 2], [-3, 3]])
        self.W1 = np.array([[-2, 2], [-3, 3]])
        self.W2 = np.array([[2], [-1.5]])
        self.b1 = np.array([[-1], [1]])
        self.b2 = np.array([[-1]])

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -100, 100)))

    def sigmoid_derivative(self, x):
        sig = self.sigmoid(x)
        return sig * (1 - sig)

    def forward(self, X):
        z0 = self.W0.T @ X
        a0 = self.sigmoid(z0)
        z1 = self.W1.T @ a0 + self.b1
        a1 = self.sigmoid(z1)
        z2 = self.W2.T @ a1 + self.b2
        a2 = z2
        print(f'A2: {a2}')
        return {'z0': z0, 'a0': a0, 'z1': z1, 'a1': a1, 'z2': z2, 'a2': a2}

    def backpropagate(self, X, y, cache):
        dz2 = cache['a2'] - y
        dW2 = dz2 @ cache['a1'].T
        db2 = dz2

        dz1 = (self.W2 @ dz2) * self.sigmoid_derivative(cache['z1'])
        dW1 = dz1 @ cache['a0'].T
        db1 = dz1

        dz0 = (self.W1.T @ dz1) * self.sigmoid_derivative(cache['z0'])
        dW0 = dz0 @ X.T
        db0 = dz0

        return {'W2': dW2, 'b2': db2, 'W1': dW1, 'b1': db1, 'W0': dW0, 'b0': db0}