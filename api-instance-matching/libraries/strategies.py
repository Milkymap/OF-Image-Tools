import cv2 
import numpy as np 
import torch as th
import torchvision as tv 
from torchvision import transforms as T 

from os import path
from glob import glob 

def to_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def th2cv(th_image):
    red, green, blue = th_image.numpy()
    return cv2.merge((blue, green, red))

def cv2th(cv_image):
    blue, green, red = cv2.split(cv_image)
    return th.as_tensor(np.stack([red, green, blue]))

def th2pl(th_image):
    return T.ToPILImage()(th_image)

def pl2th(pl_image):
    return T.ToTensor()(pl_image)

def cv2pl(cv_image):
    return th2pl(cv2th(cv_image))

def pl2cv(pl_image):
    return th2cv(pl2th(pl_image))

def read_image(path2image, size=None):
    cv_image = cv2.imread(path2image, cv2.IMREAD_COLOR)
    if size is not None:
        return cv2.resize(cv_image, size, interpolation=cv2.INTER_CUBIC)
    return cv_image

def save_image(cv_image, path2location):
    cv2.imwrite(path2location, cv_image)

def pull_files(target_location, extension='*'):
    return glob(path.join(target_location, extension))

def to_grid(batch_images, nb_rows=8, padding=10, normalize=True):
	grid_images = tv.utils.make_grid(batch_images, nrow=nb_rows, padding=padding, normalize=normalize)
	return grid_images





