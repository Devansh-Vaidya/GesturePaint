import tkinter as tk
import cv2
from PIL import Image, ImageTk


def update_canvas():
    pass


def video_capture():
    _, frame = vid.read()
    opencv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    captured_image = Image.fromarray(opencv_image)
    photo_image = ImageTk.PhotoImage(image=captured_image)

    # Displaying photoimage in the label
    label_widget.photo_image = photo_image

    # Configure image in the label
    label_widget.configure(image=photo_image)

    # Repeat the same process after every 10 seconds
    label_widget.after(10, video_capture)


def create_canvas():
    canvas = tk.Canvas(app, bg="white", cursor="cross", width=width, height=height)
    canvas.grid(row=0, column=1)


def init():
    global vid, width, height, app, label_widget, canvas
    vid = cv2.VideoCapture(0)

    # Width and height of the video
    width, height = 80, 60

    # Set the width and height
    vid.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    vid.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    # Get the width and height
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Create a GUI app
    app = tk.Tk()

    # Bind the app with Escape keyboard to quit app whenever pressed
    app.bind('<Escape>', lambda e: app.quit())

    # Configure grid to resize proportionally
    app.grid_rowconfigure(0, weight=1)
    app.grid_columnconfigure(0, weight=1)
    app.grid_columnconfigure(1, weight=1)

    # Create a label and display it on app
    label_widget = tk.Label(app)
    label_widget.grid(row=0, column=0, sticky="nsew")


def main():
    init()
    create_canvas()
    video_capture()
    app.mainloop()


if __name__ == "__main__":
    main()
