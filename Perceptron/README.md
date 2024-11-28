To run this, you will first create either the standard perceptron or voted perceptron using the line

StandardPerceptron(iterations=epoch)
change standard to voted if you want to using voting.
The default number of iterations is 10, but you can change the number of epocs using the iterations parameter.

You will then call train on the dataset using x and y, representing your features and labels. You can then run the various helper methods, including the predict_all and calculate_error_rate to calculate various values of the model.