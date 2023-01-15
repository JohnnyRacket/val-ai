from cv2 import COLOR_BGR2GRAY, resize, cvtColor, imread, imwrite
import os
import re

def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(data, key=alphanum_key)

def resize_and_grayscale_img(img):
    # downsize image to 1/4
    img = resize(img, (0, 0), fx=0.5, fy=0.5)
    # convert to grayscale
    # img = cvtColor(img, COLOR_BGR2GRAY)

    return img

def load_images_from_folder(folder):
    images = []
    for filename in sorted_alphanumeric(os.listdir(folder)):
        img = imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return images

os.chdir(r'C:\Users\John\Videos')
imgs = load_images_from_folder(r'picsv22')

for index, img in enumerate(imgs):
    img = resize_and_grayscale_img(img)
    imwrite((r'picsv2-3\img' + str(index + 1) + '.jpg'), img)

