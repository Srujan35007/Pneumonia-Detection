__author__ = 'Suddala Srujan'


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



IMG_HEIGHT, IMG_WIDTH = 50, 50
N_C = [1, 256, 256, 32, 10]
N_K = [5, 4, 3, 2]
Max_mp = max(N_K)
N_HL = [512, 128]
N_OUTPUTS = 2


def flatten_shape(k):
    prod = 1
    for i in range(len(k)):
        prod = prod*k[i]
    return prod
    

def get_maxpool_params(temp):
    def fit(x,a,b,c,d):
        return a*(x**3)+b*(x**2)+c*(x**1)+d
    y = N_K
    x = [i for i in range(len(y))]
    params, foo = curve_fit(fit, x, y)
    r_squared = []
    for i in range(len(temp)):
        sum_ = 0
        for j in range(len(N_K)):
            sum_ = sum_ + (temp[i][j]-fit(j, *params))**2
        r_squared.append(sum_)
    return temp[r_squared.index(min(r_squared))]


class Net(nn.Module):

    def __init__(self):
        super(Net,self).__init__()
        flag = False
        temp = []
        
        print('Configuring Hyper-parameters.')
        for a in tqdm(range(1,Max_mp+1,1)):
            for b in range(1,Max_mp+1-1,1):
                for c in range(1,Max_mp+1-2,1):
                    for d in range(1,Max_mp+1-3,1):
                        try:
                            bef = time.time()
                            self.conv1 = nn.Conv2d(N_C[0], N_C[1], N_K[0])
                            self.conv2 = nn.Conv2d(N_C[1], N_C[2], N_K[1])
                            self.conv3 = nn.Conv2d(N_C[2], N_C[3], N_K[2])
                            self.conv4 = nn.Conv2d(N_C[3], N_C[4], N_K[3])
                            t = torch.rand(IMG_HEIGHT, IMG_WIDTH)
                            t = t.view(-1,1,IMG_HEIGHT,IMG_WIDTH)
                            t = F.max_pool2d(self.conv1(t),(a,a))
                            t = F.max_pool2d(self.conv2(t),(b,b))
                            t = F.max_pool2d(self.conv3(t),(c,c))
                            t = F.max_pool2d(self.conv4(t),(d,d))
                            temp.append([a,b,c,d])
                            aft = time.time()
                            print('Each pass takes {} seconds'.format(aft-bef))
                        except:
                            pass
                            
        #print(temp)
        self.mp = get_maxpool_params(temp)
        print(self.mp)
        self.conv1 = nn.Conv2d(N_C[0], N_C[1], N_K[0])
        self.conv2 = nn.Conv2d(N_C[1], N_C[2], N_K[1])
        self.conv3 = nn.Conv2d(N_C[2], N_C[3], N_K[2])
        self.conv4 = nn.Conv2d(N_C[3], N_C[4], N_K[3])
        print('ConvNet created.')
        if flag is False:
            t = torch.rand(IMG_HEIGHT,IMG_WIDTH)
            t = t.view(-1,1,IMG_HEIGHT,IMG_WIDTH)
            t = F.max_pool2d(self.conv1(t),(self.mp[0],self.mp[0]))
            t = F.max_pool2d(self.conv2(t),(self.mp[1],self.mp[1]))
            t = F.max_pool2d(self.conv3(t),(self.mp[2],self.mp[2]))
            t = F.max_pool2d(self.conv4(t),(self.mp[3],self.mp[3]))
            flag = True
            t_shape = t.shape
        self.Flattened_input_shape = flatten_shape(t_shape)
        self.fc1 = nn.Linear(self.Flattened_input_shape, N_HL[0])
        self.fc2 = nn.Linear(N_HL[0], N_HL[1])
        self.fc3 = nn.Linear(N_HL[1], N_OUTPUTS)
        print('LinearNet added.')

    def forward(self,x):
        x = x.view(-1,1,IMG_HEIGHT,IMG_WIDTH)
        x = F.max_pool2d(self.conv1(x),(self.mp[0],self.mp[0]))
        x = F.max_pool2d(self.conv2(x),(self.mp[1],self.mp[1]))
        x = F.max_pool2d(self.conv3(x),(self.mp[2],self.mp[2]))
        x = F.max_pool2d(self.conv4(x),(self.mp[3],self.mp[3]))
        x = x.view(-1,self.Flattened_input_shape)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.log_softmax(self.fc3(x), dim = 1)
        return x

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
cpu = torch.device('cpu')
net = Net().to(device)


net1 = pickle.load(open('F:\\Srujan\\MyProjects\\Pneumonia\\Trained\\Pneumonia_trained_500.pickle','rb'))
net2 = pickle.load(open('F:\\Srujan\\MyProjects\\Pneumonia\\Trained\\Pneumonia500v2.pickle','rb'))
net3 = pickle.load(open('F:\\Srujan\\MyProjects\\Pneumonia\\Trained\\Pneumonia500v3.pickle','rb'))
net4 = pickle.load(open('F:\\Srujan\\MyProjects\\Pneumonia\\Trained\\Pneumonia500v4.pickle','rb'))

path = 'F:\\Srujan\\MyProjects\\Pneumonia'
filenames = [f'a ({i+1})' for i in range(234)] + [f'b ({i+1})' for i in range(390)]
random.shuffle(filenames)
classes = ['Normal', 'Pneumonia']
actual = ''
correct1,correct2,correct3,correct4 = 0,0,0,0
total = 0
with torch.no_grad():
    img = cv2.imread(f'{path}\\test7.jpg')
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resize = cv2.resize(img_gray, (500,500))
    X = torch.tensor(np.array(resize/255, dtype = np.float32))
    out1 = classes[torch.argmax(net1(X))]
    out2 = classes[torch.argmax(net2(X))]
    out3 = classes[torch.argmax(net3(X))]
    out4 = classes[torch.argmax(net4(X))]
print(out2,out3,out1,out4)        
