from cv2 import COLOR_BGR2GRAY, resize, cvtColor
import cv2

def resize_and_grayscale_img(img):
    # downsize image to 1/4
    img = resize(img, (0, 0), fx=0.5, fy=0.5)
    # convert to grayscale
    img = cvtColor(img, COLOR_BGR2GRAY)

    return img

def resize_img(img):
    # downsize image to 1/4
    img = resize(img, (0, 0), fx=0.5, fy=0.5)
    img = cvtColor(img, cv2.COLOR_BGRA2BGR)

    return img