import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import matplotlib.pyplot as plt

train_file_path = 'bank-note/train.csv'
test_file_path = 'bank-note/test.csv'
training_data = pd.read_csv(train_file_path)
testing_data = pd.read_csv(test_file_path)

training_x = training_data.iloc[:, :-1].values
testing_x = testing_data.iloc[:, :-1].values
training_y = training_data.iloc[:, -1].map({0: -1, 1: 1}).values
testing_y = testing_data.iloc[:, -1].map({0: -1, 1: 1}).values

training_x_tensor = torch.tensor(training_x, dtype=torch.float32)
training_y_tensor = torch.tensor(training_y, dtype=torch.float32).view(-1, 1)
testing_x_tensor = torch.tensor(testing_x, dtype=torch.float32)
testing_y_tensor = torch.tensor(testing_y, dtype=torch.float32).view(-1, 1)

def initialize_weights_xavier(layer):
    if isinstance(layer, nn.Linear):
        nn.init.xavier_uniform_(layer.weight)

def initialize_weights_he(layer):
    if isinstance(layer, nn.Linear):
        nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')

def create_model(input_size, hidden_layer_size, depth, activation_fn, init_fn):
    layers = []
    in_features = input_size
    # Create layers of hidden size, with activation function between
    for _ in range(depth):
        layer = nn.Linear(in_features, hidden_layer_size)
        layers.append(layer)
        layers.append(activation_fn())
        in_features = hidden_layer_size
    layers.append(nn.Linear(in_features, 1))
    model = nn.Sequential(*layers)
    model.apply(init_fn)
    return model

def train_and_evaluate_model(model, training_x, training_y, testing_x, testing_y, epochs, learning_rate):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    training_errors = []
    training_error_rates = []
    testing_error_rate = None

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(training_x)
        loss = criterion(outputs, training_y)
        loss.backward()
        optimizer.step()
        training_predictions = torch.sign(outputs)
        incorrect_training_predictions = (training_predictions.view(-1) != training_y.view(-1)).sum()
        training_error_rate = incorrect_training_predictions / len(training_y)
        training_errors.append(loss.item())
        training_error_rates.append(training_error_rate)

    model.eval()
    with torch.no_grad():
        testing_outputs = model(testing_x)
        testing_predictions = torch.sign(testing_outputs)
        incorrect_testing_predictions = (testing_predictions.view(-1) != testing_y.view(-1)).sum()

    testing_error_rate = incorrect_testing_predictions / len(testing_y)
    return training_errors, training_error_rates, testing_error_rate

# Hyperparameters and configuration
activation_functions = [(nn.Tanh, initialize_weights_xavier), (nn.ReLU, initialize_weights_he)]
depths = [3, 5, 9]
layer_sizes = [5, 10, 25, 50, 100]
epochs = 50
learning_rate = 0.001
input_size = training_x.shape[1]
results = {}

for activation_fn, init_fn in activation_functions:
    for depth in depths:
        for layer_size in layer_sizes:
            model = create_model(input_size, layer_size, depth, activation_fn, init_fn)
            training_errors, training_error_rates, testing_error_rate = train_and_evaluate_model(model, training_x_tensor, training_y_tensor, testing_x_tensor, testing_y_tensor, epochs, learning_rate)

            # Save configuration with related data
            config = f'{activation_fn.__name__}, Depth {depth}, Layer Size {layer_size}'
            results[config] = {'training_errors': training_errors, 'training_error_rates': training_error_rates, 'testing_error_rate': testing_error_rate}

# Print results
for config, metrics in results.items():
    print(f'{config}: Error rate for testing and training: {metrics["testing_error_rate"]:.4f}, {metrics["training_error_rates"][-1]:.4f}')
