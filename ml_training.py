import json

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

# Dictionary to map the labels to integers
classifier_set = {0: 'black', 1: 'blue', 2: 'draw', 3: 'erase', 4: 'green', 5: 'red'}


def extract_data_labels(data_labels, data, labels):
    """
    Extract the data and labels from the dictionary.
    Args:
        data_labels (dict): Dictionary containing the data and labels.
        data (numpy.ndarray): Data to be extracted.
        labels (numpy.ndarray): Labels to be extracted.

    Returns:
        data (numpy.ndarray): Extracted data.
        labels (numpy.ndarray): Extracted labels.
    """
    for key, value in data_labels.items():
        # Convert string of list to list
        key = key.strip('][').split(', ')

        # Convert string to float
        key = [float(i) for i in key]

        data.append(key)
        labels.append(value)
    data = np.array(data)
    min = np.min(data)
    max = np.max(data)

    # Normalize the data
    data = (data - min) / (max - min)

    df = pd.DataFrame(data={'labels': labels})

    # Print unique labels with corresponding indices
    categorical_dummies = pd.get_dummies(df['labels'])
    print(categorical_dummies)

    labels = np.asarray(categorical_dummies)

    return data, labels


def create_model():
    """
    Create a model for the gesture paint application.
    Returns:
        tensorflow.python.keras.engine.sequential.Sequential: Model for the gesture paint application.
    """
    model = Sequential(
        [
            Dense(96, input_shape=(42,), activation="relu"),
            Dense(48, activation="relu"),
            Dense(24, activation="relu"),
            Dense(12, activation="relu"),
            Dense(6, activation="softmax")
        ]
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    model.summary()
    return model


def train_model(model, data, labels, batch_size=32, epochs=100):
    """
    Train the model.
    Args:
        model (tensorflow.python.keras.engine.sequential.Sequential): Model to be trained.
        data (numpy.ndarray): Data to be used for training.
        labels (numpy.ndarray): Labels to be used for training.
        batch_size (int, optional): Batch size of the model. Defaults to 32.
        epochs (int, optional): Number of epochs for training. Defaults to 100.

    Returns:
        tensorflow.python.keras.engine.training.Model: Training history of the model.
        tensorflow.python.keras.engine.sequential.Sequential: Trained model.
    """
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.01, shuffle=True, stratify=labels)
    training_history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)
    score = model.evaluate(x_test, y_test, verbose=0)
    print("Test loss:", score[0])
    print("Test accuracy:", score[1])
    return training_history, model


def plot_learning_curve(training):
    """
    Plot the learning curve.
    Args:
        training (tensorflow.python.keras.engine.training.Model): Training history of the model.
    """
    plt.plot(training.history['loss'])
    # plt.plot(training.history['val_loss'])
    plt.title('Learning Curve')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


def main():
    """
    Main function to train the model.
    """
    # Read the data
    with open('data.json', 'r') as file:
        data_labels = json.load(file)

    # Convert the labels to one-hot encoded vectors
    data = []
    labels = []
    data, labels = extract_data_labels(data_labels, data, labels)

    # Train the model
    model = create_model()
    training, model = train_model(model, data, labels, batch_size=8, epochs=100)
    plot_learning_curve(training)

    # Save the model
    model.save('gesture_paint_model.keras')


if __name__ == '__main__':
    main()
