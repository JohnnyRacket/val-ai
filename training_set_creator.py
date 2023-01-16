import cv2
import os
import re
import numpy as np
from filter_images import direction_classifier_image_filter

def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(data, key=alphanum_key)

def load_images_from_folder(folder):
    images = []
    for filename in sorted_alphanumeric(os.listdir(folder)):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return images

# os.chdir(r'C:\Users\John\Videos')
imgs = load_images_from_folder(r'direction_classifier\direction_classifier_imgs')

os.chdir(r'C:\Users\John\Videos')
for index, img in enumerate(imgs):
    img = direction_classifier_image_filter(img)
    cv2.imwrite((r'picsv2-3\img' + str(index + 1) + '.png'), img)

