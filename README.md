# GesturePaint

GesturePaint is an innovative Python application designed to enable users to express themselves through hand gestures on a digital canvas. The application seamlessly integrates machine learning methodologies, notably MediaPipe for precise handmark recognition and Multi-Layer Perceptron for robust gesture classification. OpenCV is used for real-time video feed capture from the webcam, coupled with the versatility of the Tkinter GUI for the user interface.

## Installation

1. Clone the repository to your local machine:

```bash
git clone https://github.com/Devansh-Vaidya/GesturePaint.git
```

2. Navigate to the project directory:

```bash
cd GesturePaint
```

3. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Data Preprocessing

The dataset was prepocessed in the file `data_preprocessing.py`. The preprocessing includes:

- Augmenting the data randomly to increase the size of the dataset. This is only required when the dataset is small.
- Alternatively, you can use your own dataset and store it in `./GesturePaint/data/train/` folder. Your folder structure should look like:
    - data
        - train
            - black
                - black_1.jpg
                - ...
            - blue
            - draw
            - erase
            - green
            - red
- Creating a dictionary representing the image data and labels, which is saved in the `data.json` file.

We do not need to run this file as the `data.json` file is already available.

To run the data preprocessing, execute the following command:

```bash
python data_preprocessing.py
```

## Model Training

Model is created and trained inside the file `ml_training.py`. The file includes:

- Loading the data from the `data.json` file.
- Creating a Multi-Layer Perceptron  (MLP) model using Keras.
- Compiling the model with the Adam optimizer and categorical crossentropy loss.
- Training the model on the data.

The model `gesture_paint_model.keras` is available and can be used directly for prediction.

To run the training, execute the following command:

```bash
python ml_training.py
```

## Inference

To run the application, execute the following command:

```bash
python main.py
```

<img width="1404" alt="GesturePaintDemo" src="https://github.com/Devansh-Vaidya/GesturePaint/assets/99159017/5bdc8d4b-338c-4baa-a25f-36e24104a78c">

## Usage

1. Upon running the application, you will be greeted with a window displaying the webcam feed and a blank white canvas.
2. The application recognizes the following gestures:
    - Index finger and thumb pinched with other fingers extended: Draw on the canvas.
    - Index Finger Extended: Change brush colour to red.
    - Victory Sign: Change the brush colour to green.
    - Three fingers extended: Change the brush colour to blue.
    - Four fingers extended: Change the brush colour to black.
    - Fist: Erase the drawing on the canvas.
3. Close the application window to exit.

## Dataset

The dataset used in this project was meticulously curated and created by me. It comprises images of hand gestures under various lighting conditions and backgrounds. The dataset plays a crucial role in training and evaluating the machine learning models implemented in this project, ensuring robust performance and accuracy. For inquiries regarding the dataset or potential collaborations, please contact me.

## Acknowledgements

I would like to extend my gratitude to the following individuals and institutions for their invaluable contributions to this project:

- **[Concordia University](https://www.concordia.ca/)**: I am thankful to my alma mater for providing me with the essential courses and knowledge that laid the foundation for this project.

- **[Coursera](https://www.coursera.org/)**: Special thanks to Coursera for offering the course "Machine Learning Specialization" by DeepLearning.AI, which I completed as part of my learning journey.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
