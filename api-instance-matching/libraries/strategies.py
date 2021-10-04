import cv2 
import numpy as np

from os import path
from glob import glob 

def to_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def read_image(path2image, size=None):
    cv_image = cv2.imread(path2image, cv2.IMREAD_COLOR)
    if size is not None:
        return cv2.resize(cv_image, size, interpolation=cv2.INTER_CUBIC)
    return cv_image

def save_image(cv_image, path2location):
    cv2.imwrite(path2location, cv_image)

def pull_files(target_location, extension='*'):
    return glob(path.join(target_location, extension))





