from PIL import Image
import numpy as np
import os


def load_images(cats_path, dogs_path):
    X = []
    y = []

    # Cats
    for filename in os.listdir(cats_path):
        if filename.endswith(".jpg"):
            path = os.path.join(cats_path, filename)

            image = Image.open(path)
            image = image.convert("RGB")
            image = image.resize((64, 64))
            image = np.array(image)

            X.append(image)
            y.append(0)  # cat

    # Dogs
    for filename in os.listdir(dogs_path):
        if filename.endswith(".jpg"):
            path = os.path.join(dogs_path, filename)

            image = Image.open(path)
            image = image.convert("RGB")
            image = image.resize((64, 64))
            image = np.array(image)

            X.append(image)
            y.append(1)  # dog

    X = np.array(X)
    y = np.array(y)

    # Normalize pixels
    X = X / 255.0

    return X, y


# Paths
train_cats_path = "../data/train/cats"
train_dogs_path = "../data/train/dogs"

test_cats_path = "../data/test/cats"
test_dogs_path = "../data/test/dogs"


# Load datasets
X_train, y_train = load_images(
    train_cats_path,
    train_dogs_path
)

X_test, y_test = load_images(
    test_cats_path,
    test_dogs_path
)


# Information
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)

print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)

print("Pixel minimum:", X_train.min())
print("Pixel maximum:", X_train.max())

print("Training cats:", np.sum(y_train == 0))
print("Training dogs:", np.sum(y_train == 1))

print("Testing cats:", np.sum(y_test == 0))
print("Testing dogs:", np.sum(y_test == 1))

#flatening
X_train = X_train.reshape(X_train.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)

print("X_train flattened:", X_train.shape)
print("X_test flattened:", X_test.shape)