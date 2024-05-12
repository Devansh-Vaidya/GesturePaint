import tkinter as tk
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import time
import PIL
from PIL import Image, ImageTk


class App:
    def __init__(self, window, window_title, video_source=0):
        self.photo = None
        self.window = window
        self.window.title(window_title)
        self.video_source = video_source

        # open video source (by default this will try to open the computer webcam)
        self.vid = MyVideoCapture(self.video_source)

        self.tools = tk.Frame(window)
        self.tools.grid(row=0, column=1)

        # Labels for brush and eraser
        self.brush_label = tk.Label(self.tools, text="Brush", bg="black")
        self.brush_label.grid(row=0, column=0)

        self.eraser_label = tk.Label(self.tools, text="Eraser", bg="black")
        self.eraser_label.grid(row=0, column=1)

        # Create a canvas that can fit the above video source size
        self.video_canvas = tk.Canvas(window, width=self.vid.width, height=self.vid.height)
        self.video_canvas.grid(row=1, column=0)

        # Create a drawing canvas
        self.canvas = tk.Canvas(window, bg="white", cursor="cross", width=self.vid.width, height=self.vid.height)
        self.canvas.grid(row=1, column=1)

        # MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

        # After it is called once, the update method will be automatically called every delay milliseconds
        self.delay = 10
        self.update()

        self.window.mainloop()

    def snapshot(self):
        # Get a frame from the video source
        ret, frame = self.vid.get_frame()

        if ret:
            cv2.imwrite("frame-" + time.strftime("%d-%m-%Y-%H-%M-%S") + ".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    def update(self):
        # Get a frame from the video source
        ret, frame = self.vid.get_frame()

        if ret:
            self.photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(frame))
            self.video_canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

            # Process the image
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(frame_rgb)

            # Process the image
            normalized_data, x_points, y_points = [], [], []
            H, W, _ = frame.shape

            results = self.hands.process(frame_rgb)

            # Predict the sign and draw on the canvas
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(
                        frame,  # image to draw
                        hand_landmarks,  # model output
                        self.mp_hands.HAND_CONNECTIONS,  # hand connections
                        self.mp_drawing_styles.get_default_hand_landmarks_style(),
                        self.mp_drawing_styles.get_default_hand_connections_style())

                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y

                        x_points.append(x)
                        y_points.append(y)

                        normalized_data.append(x - min(x_points))
                        normalized_data.append(y - min(y_points))

                data = np.asarray(normalized_data)
                # if data.shape[0] == 42:
                #     # data = data.reshape(1, 42)
                #     # result = model.predict(data)
                #     # if the result is draw then call the draw function else call the eraser function
                #     # if np.argmax(result) == 0:
                #     x1 = hand_landmarks.landmark[8].x
                #     y1 = hand_landmarks.landmark[8].y
                #
                #     x2 = hand_landmarks.landmark[4].x
                #     y2 = hand_landmarks.landmark[4].y
                #
                #     # Average of the two points
                #     x_point = (x1 + x2) / 2
                #     y_point = (y1 + y2) / 2

                # draw(frame, x_point, y_point)
                # elif np.argmax(result) == 1:
                #     erase(frame, x_points, y_points)

                normalized_data = []
        self.window.after(self.delay, self.update)


def draw(frame, x_point, y_point):
    # Draw on the canvas
    x = int(x_point) - 10
    y = int(y_point) - 10

    cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)


class MyVideoCapture:
    def __init__(self, video_source=0):
        # Open the video source
        self.vid = cv2.VideoCapture(video_source)
        if not self.vid.isOpened():
            raise ValueError("Unable to open video source", video_source)

        # Get video source width and height
        self.width = 800
        self.height = 600

    def get_frame(self):
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
        if self.vid.isOpened():
            self.vid.release()


def main():
    # Create a window and pass it to the Application object
    App(tk.Tk(), "Tkinter and OpenCV")


if __name__ == "__main__":
    main()
