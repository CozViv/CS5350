import numpy as np

class SVM:
    def __init__(self, C, schedule, gamma, a=1, max_epochs=100):
        self.C = C
        self.schedule = schedule
        self.gamma = gamma
        self.a = a
        self.max_epochs = max_epochs
        self.objective_function_values = []
        self.w = None
        self.b = None

    def train(self, x, y):
        size_samples, size_features = x.shape

        self.w = np.zeros(size_features)
        self.b = 0

        self.objective_function_values = []

        for epoch in range(self.max_epochs):
            rand_indices = np.random.permutation(size_samples)
            x, y = x[rand_indices], y[rand_indices]

            for t, (xi, yi) in enumerate(zip(x, y)):
                gamma_temp = self.schedule(self.gamma, self.a, t + epoch * size_samples)

                if yi * (np.dot(self.w, xi) + self.b) <= 1:
                    self.w = (1 - gamma_temp / self.C) * self.w + gamma_temp * size_samples * self.C * yi * xi
                    self.b += gamma_temp * self.C * yi
                else:
                    self.w = (1 - gamma_temp / self.C) * self.w

                # To test curve
                hinge_loss = np.maximum(0, 1 - y * (np.dot(x, self.w) + self.b)).sum()
                objective = (1 / 2) * np.dot(self.w, self.w) + self.C * hinge_loss
                self.objective_function_values.append(objective)

    def predict(self, x):
        return np.sign(np.dot(x, self.w) + self.b)

    def evaluate(self, x, y):
        predictions = self.predict(x)
        return np.mean(predictions != y)
