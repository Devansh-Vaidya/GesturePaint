import cv2
import os
import json
import numpy as np
from mediapipe import solutions
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def process_data(path):
    """
    Process the data in the directory.
    Args:
        path (str): Path to the directory containing the data.

    Returns:
        dict: Dictionary containing the data and labels.
    """
    data_label_dict = {}
    hands = solutions.hands.Hands(static_image_mode=True, max_num_hands=2)
    for directory in os.listdir(path):
        directory_path = os.path.join(path, directory)
        if os.path.isdir(directory_path):
            process_images(directory_path, hands, directory, data_label_dict)
    return data_label_dict


def process_images(path, hands, directory="", data_label_dict=None):
    """
    Process the images in the directory.
    Args:
        path (str): Path to the directory containing the images.
        hands (mediapipe.solutions.hands.Hands): Hands object.
        directory (str, optional): Directory name. Defaults to "".
        data_label_dict (dict, optional): Dictionary containing the data and labels. Defaults to None.
    """
    for img in os.listdir(path):
        img_path = os.path.join(path, img)

        # Check if the file is an image
        if not img_path.endswith('.jpeg'):
            continue

        img_rgb = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        landmarks = hands.process(img_rgb)
        process_img(landmarks, directory, data_label_dict)


def process_img(landmarks, directory, data_label_dict):
    """
    Process the landmarks of the image.
    Args:
        landmarks (mediapipe.solutions.hands.Hands.process): Landmarks of the image.
        directory (str): Directory name.
        data_label_dict (dict): Dictionary containing the data and labels.
    """
    x_points = []
    y_points = []
    normalized_data = []
    if landmarks.multi_hand_landmarks:
        for hand in landmarks.multi_hand_landmarks:
            for i in range(len(hand.landmark)):
                x = hand.landmark[i].x
                y = hand.landmark[i].y
                x_points.append(x)
                y_points.append(y)
            for i in range(len(hand.landmark)):
                x = hand.landmark[i].x
                y = hand.landmark[i].y
                normalized_data.append(x - min(x_points))
                normalized_data.append(y - min(y_points))

            data_label_dict[str(normalized_data)] = directory
            normalized_data = []


def augment_data(path):
    """
    Augment the data in the directory.
    Args:
        path (str): Path to the directory containing the data.
    """
    subdirs = [x[0] for x in os.walk(path)]
    for subdir in subdirs:
        img_paths = [os.path.join(subdir, f) for f in os.listdir(subdir) if f.endswith('.jpeg')]
        for img_path in img_paths:
            if img_path.endswith('.jpeg'):
                augment_image(img_path)


def augment_image(img_path):
    """
    Augment the image.
    Args:
        img_path (str): Path to the image.
    """
    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    img = np.expand_dims(img, axis=0)
    img_dir_path = img_path.split('.')[1]
    datagen = ImageDataGenerator(
        rotation_range=10,
        zoom_range=0.1,
        horizontal_flip=True
    )

    # Iterator
    aug_iter = datagen.flow(img, batch_size=1)

    # Generate batch of images
    for i in range(10):
        # Convert to unsigned integers
        image = next(aug_iter)[0].astype('uint8')
        cv2.imwrite(f'.{img_dir_path}_{i}.jpeg', cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


def save_data(data_label_dict):
    """
    Save the data to a JSON file.
    Args:
        data_label_dict (dict): Dictionary containing the data and labels.
    """
    with open('data.json', 'w') as file:
        json.dump(data_label_dict, file)


def main():
    """
    Main function to process the data.
    """
    # augment_data('./data/train')
    data_label_dict = process_data('./data/train')
    save_data(data_label_dict)


if __name__ == '__main__':
    main()
