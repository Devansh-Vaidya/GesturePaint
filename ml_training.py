import json

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split


def extract_data_labels(data_labels, data, labels):
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
    print(df['labels'].value_counts())

    categorical_dummies = pd.get_dummies(df['labels'])
    labels = np.asarray(categorical_dummies)
    return data, labels


def create_model():
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


# Train the model and return the training history and the model
def train_model(model, data, labels, batch_size=32, epochs=100):
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.1, shuffle=True, stratify=labels)
    training_history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.2)
    score = model.evaluate(x_test, y_test, verbose=0)
    print("Test loss:", score[0])
    print("Test accuracy:", score[1])
    return training_history, model


# Plot the learning curve of the model
def plot_learning_curve(training):
    plt.plot(training.history['loss'])
    plt.plot(training.history['val_loss'])
    plt.title('Learning Curve')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


def main():
    # Read the data
    with open('data.json', 'r') as file:
        data_labels = json.load(file)

    # Convert the labels to one-hot encoded vectors
    data = []
    labels = []
    data, labels = extract_data_labels(data_labels, data, labels)

    # Train the model
    model = create_model()
    training, model = train_model(model, data, labels, batch_size=8, epochs=200)
    plot_learning_curve(training)

    # Save the model
    model.save('gesture_paint_model.h5')


if __name__ == '__main__':
    main()
