import cv2 
import numpy as np 
import torch as th

from torchvision import transforms as T 
from PIL import Image 

from os import path
from glob import glob 

def th2cv(th_image):
    red, green, blue = th_image.numpy()
    return cv2.merge((blue, green, red))

def cv2th(cv_image):
    blue, green, red = cv2.split(cv_image)
    return th.as_tensor(np.stack([red, green, blue]))

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

def prepare_image(th_image):
    normalied_th_image = th_image / th.max(th_image)
    return T.Compose([
        T.Resize((256, 256)),
        T.CenterCrop((224, 224)),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])(normalied_th_image)







