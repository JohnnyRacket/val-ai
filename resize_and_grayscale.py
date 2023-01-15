from cv2 import COLOR_BGR2GRAY, resize, cvtColor

def resize_and_grayscale_img(img):
    # downsize image to 1/4
    img = resize(img, (0, 0), fx=0.25, fy=0.25)
    # convert to grayscale
    img = cvtColor(img, COLOR_BGR2GRAY)

    return img

def resize_img(img):
    # downsize image to 1/4
    img = resize(img, (0, 0), fx=0.25, fy=0.25)

    return img