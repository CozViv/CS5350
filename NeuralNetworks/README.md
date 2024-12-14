To run this, you will first create a NeuralNetwork using the initialization function, inputting the number of inputs and the
desired size of the hidden layers.

NeuralNetwork(input_size=..., hidden_layer_size=...)
Use the above line to initialize it

Train the model using. This model uses a schedule, where the learning rate is gamma0/(1+gamma0/d)*t.
Adjust the inital gamma and d to learn models depening on size.
nn.train(training_data, training_labels, epochs=epochs, gamma0=.1, d=0.1)

Train returns an array of errors after every epoch, use this to see if the model has converged and is training properly.

Use predict_all on your testing data to get a list of predictions using the model