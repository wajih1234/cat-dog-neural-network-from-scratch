import numpy as np

from data import X_train, y_train, X_test, y_test
from model import NeuralNetwork


# Make the random initialization reproducible
np.random.seed(42)


# Create the neural network
model = NeuralNetwork(
    input_size=X_train.shape[1]
)


# Train the model
model.train(
    X_train,
    y_train,
    epochs=1000,
    learning_rate=0.001
)

train_predictions = model.predict(X_train)
test_predictions = model.predict(X_test)

train_accuracy = np.mean(
    train_predictions == y_train.reshape(-1, 1)
)

test_accuracy = np.mean(
    test_predictions == y_test.reshape(-1, 1)
)

print("Training accuracy:", train_accuracy)
print("Test accuracy:", test_accuracy)