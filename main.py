import numpy as np
import cv2
import mediapipe as mp
import tkinter as tk
from tensorflow.keras import models
from PIL import Image, ImageTk
from ml_training import classifier_set


class App:
    """
    Class that creates the GUI for the application, handles the video feed, and processes the image.
    """

    def __init__(self, window: tk.Tk, window_title: str, video_source: int = 0):
        """
        Initialize the application.
        Args:
            window (tk.Tk): Tkinter window object.
            window_title (str): Title of the window.
            video_source (int, optional): Video source. Defaults to 0.
        """
        # Initialize the window and other variables
        self.photo = None
        self.window = window
        self.window.resizable(False, False)
        self.window.title(window_title)
        self.video_source = video_source
        self.model = models.load_model('gesture_paint_model.keras')

        # Open the video source
        self.vid = MyVideoCapture(self.video_source)

        # Canvas initialization
        self.fill_color = "black"
        self.tools = tk.Frame(window)
        self.tools.grid(row=0, column=1)

        # Labels for brush and eraser
        self.brush_label = tk.Label(self.tools, text="Selected: Brush (Black)")
        self.brush_label.grid(row=0, column=0)

        # Brush size for drawing and erasing
        self.brush_size = 6

        # Last selected color
        self.last_color = "black"

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
        self.delay = 1
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
            self.photo = ImageTk.PhotoImage(image=Image.fromarray(frame))
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
                    print(output, prediction)

                    if prediction == 'erase':
                        self.fill_color = "white"
                        self.brush_size = 30
                        self.brush_label.config(text="Selected: Eraser")
                        self.draw(hand_landmarks, self.brush_size)
                    else:
                        self.brush_size = 6
                        if prediction == 'draw':
                            self.fill_color = self.fill_color if self.fill_color != "white" else self.last_color
                            self.brush_label.config(text=f"Selected: Brush ({self.fill_color.capitalize()})")
                            self.draw(hand_landmarks, self.brush_size)
                        else:
                            self.brush_label.config(text=f"Selected: Brush ({self.fill_color.capitalize()})")
                            self.fill_color = prediction
                            self.last_color = prediction

        self.window.after(self.delay, self.update)

    def draw(self, hand_landmarks: mp.solutions.hands.Hands, brush_size: int = 6):
        """
        Draw on the canvas.
        Args:
            hand_landmarks (mp.solutions.hands.Hands): Hand landmarks.
            brush_size (int, optional): Brush size. Defaults to 6.
        """
        x_point = int(hand_landmarks.landmark[8].x * self.vid.width)
        y_point = int(hand_landmarks.landmark[8].y * self.vid.height)

        self.canvas.create_oval(x_point - brush_size, y_point - brush_size, x_point + brush_size, y_point + brush_size,
                                fill=self.fill_color, outline="")
        if self.fill_color == "white":
            self.canvas.create_oval(x_point - 2, y_point - 2, x_point + 2, y_point + 2,
                                    fill="red", outline="")


class MyVideoCapture:
    """
    Class that captures video frames from a video source.
    """

    def __init__(self, video_source: int = 0):
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
