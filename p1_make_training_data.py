__author__  = 'Suddala Srujan'

import time 
b = time.time()
import torch
import torchvision
from torchvision import datasets, transforms
import torch.nn.functional as F 
import torch.nn as nn 
import torch.optim as optim
import matplotlib.pyplot as plt 
import cv2 
import pickle
import random 
import numpy as np
from scipy.optimize import curve_fit
from tqdm.notebook import tqdm
import numpy as np 
import os 
from pathlib import Path
a = time.time()
print('Imports complete in {} seconds.'.format(a-b))

path = 'path_to_your_training_data_folder'
files = os.listdir(path)

Normal_label = torch.tensor([1,0])
Pneumonia_label = torch.tensor([0,1])

Normal50 = []
Pneumonia50 = []

for file in tqdm(files):
    img = cv2.imread(f'{path}\\{file}')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resize = cv2.resize(gray, (50,50))
    tnsr = torch.tensor(np.array(resize/255, dtype = np.float32))
    if 'a' in file:
        Normal50.append([tnsr, Normal_label])
    elif 'b' in file:
        Pneumonia50.append([tnsr, Pneumonia_label])
print(len(Normal50), len(Pneumonia50))

pickle.dump(Normal50, open('Normal50.pickle', 'wb'))
pickle.dump(Pneumonia50, open('Pneumonia50.pickle', 'wb'))
# You can use any other data storing modules in python 
#torch.save() is a good storing function (it stores values in .pt or .pth files)

