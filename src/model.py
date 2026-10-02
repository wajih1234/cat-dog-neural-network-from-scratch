import numpy as np


class NeuralNetwork:

    def __init__(self, input_size):
        self.weights = np.random.randn(input_size, 1) * 0.01
        self.bias = 0.0

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def forward(self, X):
        z = np.dot(X, self.weights) + self.bias
        y_pred = self.sigmoid(z)
        return y_pred

    def compute_loss(self, y_true, y_pred):
        loss = -(
            y_true * np.log(y_pred)
            + (1 - y_true) * np.log(1 - y_pred)
        )

        return np.mean(loss)

    def backward(self, X, y_true, y_pred):
        m = X.shape[0]

        dZ = y_pred - y_true

        dW = np.dot(X.T, dZ) / m

        db = np.sum(dZ) / m

        return dW, db

    def update_parameters(self, dW, db, learning_rate):
        self.weights -= learning_rate * dW
        self.bias -= learning_rate * db

    def train(self, X, y, epochs, learning_rate):
        y = y.reshape(-1, 1)

        for epoch in range(epochs):

            y_pred = self.forward(X)

            loss = self.compute_loss(y, y_pred)

            dW, db = self.backward(X, y, y_pred)

            self.update_parameters(
                dW,
                db,
                learning_rate
            )

            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss}")

    def predict(self, X):
     y_pred = self.forward(X)

     predictions = (y_pred >= 0.5).astype(int)

     return predictions