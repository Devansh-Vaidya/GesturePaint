import tkinter as tk
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import PIL
from PIL import Image, ImageTk
from ml_training import classifier_set


class App:
    """
    Class that creates the GUI for the application, handles the video feed, and processes the image.
    """

    def __init__(self, window, window_title, video_source=0):
        """
        Initialize the application.
        Args:
            window (tkinter.Tk): Tkinter window object.
            window_title (str): Title of the window.
            video_source (int, optional): Video source. Defaults to 0.
        """
        # Initialize the window and other variables
        self.photo = None
        self.window = window
        self.window.title(window_title)
        self.video_source = video_source
        self.model = tf.keras.models.load_model('gesture_paint_model.h5')

        # Open the video source
        self.vid = MyVideoCapture(self.video_source)

        # Canvas initialization
        self.fill_color = "black"
        self.tools = tk.Frame(window)
        self.tools.grid(row=0, column=1)

        # Labels for brush and eraser
        self.brush_label = tk.Label(self.tools, text="Brush", bg="black")
        self.brush_label.grid(row=0, column=0)
        self.eraser_label = tk.Label(self.tools, text="Eraser", bg="black")
        self.eraser_label.grid(row=0, column=1)

        # Create a video canvas to display the video feed
        self.video_canvas = tk.Canvas(window, width=self.vid.width, height=self.vid.height)
        self.video_canvas.grid(row=1, column=0)

        # Create a drawing canvas
        self.canvas = tk.Canvas(window, bg="white", cursor="cross", width=self.vid.width, height=self.vid.height)
        self.canvas.grid(row=1, column=1)

        # MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        # Set the delay to get a smooth video feed
        self.delay = 10
        self.update()

        self.window.mainloop()

    def update(self):
        """
        Update the video feed and process the image.
        """
        # Get a frame from the video source
        ret, frame = self.vid.get_frame()

        if ret:
            # Add the frame to the video canvas
            self.photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(frame))
            self.video_canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

            # Process the image
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(frame_rgb)

            normalized_data, x_points, y_points = [], [], []
            H, W, _ = frame.shape

            # Predict the sign and draw on the canvas
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y

                        x_points.append(x)
                        y_points.append(y)

                        normalized_data.append(x - min(x_points))
                        normalized_data.append(y - min(y_points))

                data = np.asarray(normalized_data)
                if data.shape[0] == 42:
                    data = data.reshape(1, 42)
                    output = np.argmax(self.model.predict(data))

                    prediction = classifier_set[output]
                    print(self.model.predict(data), output, prediction)

                    if prediction == 'draw':
                        self.draw(hand_landmarks, 'black', 'green', 'black')

                    elif prediction == 'erase':
                        self.draw(hand_landmarks, 'white', 'black', 'green')

                    elif prediction == 'black':
                        self.fill_color = "black"

                    elif prediction == 'red':
                        self.fill_color = "red"

                    elif prediction == 'green':
                        self.fill_color = "green"

                    elif prediction == 'blue':
                        self.fill_color = "blue"

                normalized_data = []
        self.window.after(self.delay, self.update)

    def draw(self, hand_landmarks, fill_color, brush_label_color, eraser_label_color):
        self.brush_label.config(bg=brush_label_color)
        self.eraser_label.config(bg=eraser_label_color)
        x_point = hand_landmarks.landmark[8].x
        y_point = hand_landmarks.landmark[8].y
        self.fill_color = fill_color
        radius = 6
        self.canvas.create_oval(x_point - radius, y_point - radius, x_point + radius, y_point + radius,
                                fill=self.fill_color, outline="")


class MyVideoCapture:
    """
    Class that captures video frames from a video source.
    """

    def __init__(self, video_source=0):
        """
        Initialize the video capture object.
        Args:
            video_source (int, optional): Video source. Defaults to 0.

        Raises:
            ValueError: Unable to open video source.
        """
        # Open the video source
        self.vid = cv2.VideoCapture(video_source)

        # Set the width and height of the video source
        self.vid.set(cv2.CAP_PROP_FRAME_WIDTH, 200)
        self.vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 100)

        # Get width and height of the video source
        self.width = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)

        if not self.vid.isOpened():
            raise ValueError("Unable to open video source", video_source)

    def get_frame(self):
        """
        Get a frame from the video source.
        Returns:
            bool: Boolean success flag.
            ndarray: Current frame converted to RGB.
        """
        if self.vid.isOpened():
            ret, frame = self.vid.read()
            if ret:
                # Return a boolean success flag and the current frame converted to BGR
                frame = cv2.flip(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), 1)
                return ret, frame
            else:
                return ret, None
        else:
            return None, None

    # Release the video source when the object is destroyed
    def __del__(self):
        """
        Release the video source when the object is destroyed.
        """
        if self.vid.isOpened():
            self.vid.release()


def main():
    """
    Main function to run the application.
    """
    # Create the application
    App(tk.Tk(), "Gesture Paint")


if __name__ == "__main__":
    main()
