# GesturePaint

GesturePaint is an innovative Python application designed to empower users with the ability to express themselves through hand gestures on a digital canvas. The application seamlessly integrates machine learning methodologies, notably harnessing MediaPipe for precise hand landmark recognition and Multilayer Perception for robust gesture classification. Using OpenCV for real-time video feed capture from the webcam, coupled with the versatility of Tkinter GUI, offers users a platform to unleash their creativity.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Run the Application](#run-the-application)
- [Usage](#usage)
- [Dataset](#dataset)
- [Data Preprocessing](#data-preprocessing)
- [Model Training](#model-training)
- [Credits](#credits)
- [License](#license)

## Prerequisites

- Python (version 3.11)
- pip (version 24.0)

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

Now you are all set to run the application!

## Run the Application

To run the application, execute the following command:

```bash
python main.py
```

## Usage

1. Upon running the application, you will be greeted with a window displaying the webcam feed and a canvas.
2. To draw on the canvas, simply use your hand gestures. The application recognizes the following gestures:
    - Index finger and thumb pinched with other fingers extended: Draw on the canvas.
    - Number One/Index Finger Extended: Change brush colour to red.
    - Number Two/Victory Sign: Change the brush colour to green.
    - Number Three/Three fingers extended: Change the brush colour to blue.
    - Number Four/Four fingers extended: Change the brush colour to black.
    - Fist: Erase the drawing on the canvas.
3. Close the application window to exit.

## Dataset

The dataset used in this project was meticulously curated and created by me. It comprises images of hand gestures under various lighting conditions and backgrounds. The dataset plays a crucial role in training and evaluating the machine learning models implemented in this project, ensuring robust performance and accuracy. For inquiries regarding the dataset or potential collaborations, please contact me.

## Data Preprocessing

The dataset was prepocessed in the file `data_preprocessing.py`. The preprocessing includes:

- Augmenting the data randomly to increase the size of the dataset. This is only required when the dataset is small.
- Creating a dictionary representing the image data and labels, which is saved in the `data.json` file.

We do not need to run this file as the `data.json` file is already available.

## Model Training

Model is created and trained inside the file `model_training.py`. The file includes:

- Loading the data from the `data.json` file.
- Creating a Multilayer Perceptron (MLP) model using Keras.

The model `gesture_paint_model.keras` is available and can be used directly for prediction. Hence, it is not necessary to run this file.

## Credits

I would like to extend my gratitude to the following individuals and institutions for their invaluable contributions to this project:

- **[Concordia University](https://www.concordia.ca/)**: I am thankful to my alma mater for providing me with the essential courses and knowledge that laid the foundation for this project.

- **[Coursera](https://www.coursera.org/)**: Special thanks to Coursera for offering the course "Machine Learning Specialization" by DeepLearning.AI, which I completed as part of my learning journey.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
