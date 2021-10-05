import cv2 
import numpy as np

from PIL import Image 
from os import path
from glob import glob 

def to_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def read_image(path2image, size=None):
    pl_image = Image.open(path2image).convert('RGB')
    cv_image = cv2.cvtColor(np.array(pl_image), cv2.COLOR_RGB2BGR)
    if size is not None:
        return cv2.resize(cv_image, size, interpolation=cv2.INTER_CUBIC)
    return cv_image

def save_image(cv_image, path2location):
    cv2.imwrite(path2location, cv_image)

def pull_files(target_location, extension='*'):
    return glob(path.join(target_location, extension))





