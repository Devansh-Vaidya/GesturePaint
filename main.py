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

        # After it is called once, the update method will be automatically called every delay milliseconds
        self.delay = 15
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

        self.window.after(self.delay, self.update)


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
                return (ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            else:
                return (ret, None)
        else:
            return (None, None)

    # Release the video source when the object is destroyed
    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()


def main():
    # Create a window and pass it to the Application object
    App(tk.Tk(), "Tkinter and OpenCV")


if __name__ == "__main__":
    main()
