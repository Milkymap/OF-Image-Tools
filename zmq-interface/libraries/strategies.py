import cv2 
import numpy as np 
import torch as th

from torchvision import transforms as T 
from torchvision.utils import make_grid

from os import path, read 
import glob 

def th2cv(th_image):
    red, green, blue = th_image.numpy()
    return cv2.merge((blue, green, red))

def cv2th(cv_image):
    blue, green, red = cv2.split(cv_image)
    return th.as_tensor(np.stack([red, green, blue]))

def read_image(path2image):
    cv_image = cv2.imread(path2image, cv2.IMREAD_COLOR)
    return cv_image

def save_image(cv_image, path2location):
    cv2.imwrite(path2location, cv_image)

def pull_files(target_location, extension='*'):
    return glob.glob(path.join(target_location, extension))

def load_neighbors(neighbor_paths):
    acc = []
    for npath in neighbor_paths:
        resized_tensor = T.Resize((256, 256))(cv2th(read_image(npath)))
        acc.append(resized_tensor)
    
    return th2cv(make_grid(acc, nrow=4, padding=2))




