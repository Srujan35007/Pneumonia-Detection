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

def one_hot(k, N_OUTPUTS):
    a = []
    for i in range(N_OUTPUTS):
        if i != k:
            a.append(0)
        else:
            a.append(1)
    return torch.tensor(a)


def plot_acc_loss(epochs, acc, loss): #Inputs: array of epoch_count, acc, loss
    #plt.plot(epochs, acc, linewidth = 0.8, color = 'b', label = 'val_acc')
    plt.plot(epochs, loss, linewidth = 0.8, color = 'r', label = 'val_loss')
    plt.legend()
    plt.title('Validation accuracy and loss')
    plt.show()

def Train_optimizer(val_loss_list):
    temp = val_loss_list
    val_loss_list = val_loss_list[-10:]
    if len(val_loss_list) <= 6:
        if min(val_loss_list) == val_loss_list[len(val_loss_list)-1]:
            if SAVE_MODEL is True:
                net.to(cpu)
                if Path(f'./{checkpoint_filename}').is_file():
                    os.remove(f'./{checkpoint_filename}')
                    print('Previous model removed.')
                    pickle.dump(net, open(f'./{checkpoint_filename}', 'wb'))
                    print(f'New model pickled. Reference epoch = {len(val_loss_list)}.')
                    mem_size = os.stat(f'./{checkpoint_filename}').st_size
                    print(f'Size of the model file = {mem_size//(1024**2)} MB. {(mem_size%(1024**2))/1024} KB.')
                    net.to(device)
                    
                else:
                    pickle.dump(net, open(f'./{checkpoint_filename}', 'wb'))
                    print(f'Model pickled. Reference epoch = {len(val_loss_list)}.')
                    mem_size = os.stat(f'./{checkpoint_filename}').st_size
                    print(f'Size of the model file = {mem_size//(1024**2)} MB. {(mem_size%(1024**2))/1024} KB.')
                    net.to(device)   
                    
            else:
                pass
        else:
            return False
        return True
    else:
        def new_minima_found():
            new_loss = val_loss_list[len(val_loss_list)-1]
            compare_arr = val_loss_list[:(len(val_loss_list)-1)]
            flag = False
            if new_loss < min(compare_arr):
                flag = True
                if SAVE_MODEL is True:
                    net.to(cpu)
                    if Path(f'./{checkpoint_filename}').is_file():
                        os.remove(f'./{checkpoint_filename}')
                        print('Previous model removed.')
                        print(f'New minima = {new_loss}.')
                        pickle.dump(net, open(f'./{checkpoint_filename}', 'wb'))
                        print(f'New model pickled. Reference epoch = {len(temp)}.')
                        mem_size = os.stat(f'./{checkpoint_filename}').st_size
                        print(f'Size of the model file = {mem_size//(1024**2)} MB. {(mem_size%(1024**2))/1024} KB.')
                        net.to(device)
                        
                    else:
                        pickle.dump(net, open(f'./{checkpoint_filename}', 'wb'))
                        print(f'Model pickled. Reference epoch = {len(temp)}.')
                        mem_size = os.stat(f'./{checkpoint_filename}').st_size
                        print(f'Size of the model file = {mem_size//(1024**2)} MB. {(mem_size%(1024**2))/1024} KB.')
                        net.to(device)
                        
                else:
                    pass
            else:
                pass
            return flag

        def change_in_slope_sign():
            def fit(x,a,b,c,d,e,f):
                return a*(x**5)+b*(x**4)+c*(x**3)+d*(x**2)+e*(x**1)+f

            def fit_slope(x,a,b,c,d,e,f):
                return 5*a*(x**4)+4*b*(x**3)+3*c*(x**2)+2*d*(x**1)+e

            if len(val_loss_list) <= 10:
                x = [i+1 for i in range(len(val_loss_list))] 
                y = val_loss_list
            else:
                x = [i+val_loss_list[0] for i in range(10)]
                y = val_loss_list
            params, foo = curve_fit(fit, x, y)
            x_fit = [i for i in np.arange(x[0], x[len(x)-1]+0.1, 0.1)]
            y_fit = [fit(i, *params) for i in np.arange(x[0], x[len(x)-1]+0.1, 0.1)]
            slopes = [fit_slope(i, *params) for i in np.arange(x[0], x[len(x)-1]+0.1, 0.1)]
            positive_slopes = []
            negative_slopes = []
            for i in range(len(slopes)):
                if slopes[i] > 0:
                    positive_slopes.append(slopes[i])
                elif slopes[i] < 0:
                    negative_slopes.append(slopes[i])
            if len(positive_slopes) == 0:
                return False
            else:
                if slopes.index(min(positive_slopes)) > slopes.index(max(negative_slopes)):
                    return True
                else:
                    return False
            
        def decreasing_loss():
            def fit(x,a,b,c,d,e,f):
                return a*(x**5)+b*(x**4)+c*(x**3)+d*(x**2)+e*(x**1)+f

            def fit_slope(x,a,b,c,d,e,f):
                return 5*a*(x**4)+4*b*(x**3)+3*c*(x**2)+2*d*(x**1)+e+f*0
            if len(val_loss_list) <= 10:
                x = [i+1 for i in range(len(val_loss_list))] 
                y = val_loss_list
            else:
                x = [i+len(temp)-9 for i in range(10)]
                y = val_loss_list
            params, foo = curve_fit(fit, x, y)
            slopes = [fit_slope(i, *params) for i in np.arange(x[0], x[len(x)-1]+0.1, 0.1)]
            flag = False
            for i in range(len(slopes)):
                if slopes[i] < 0:
                    flag = True
                    break
                else:
                    pass
            return flag
                

        if decreasing_loss() and change_in_slope_sign() and not new_minima_found():
            return False
        elif decreasing_loss() and change_in_slope_sign() and new_minima_found():
            return True
        elif decreasing_loss() and not change_in_slope_sign() and new_minima_found():
            return True
        elif not decreasing_loss():
            return False
        else:
            return False

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

# You actually don't need a temporary net 
# It's just a nice function that figures out the max_pool dims
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
        #print(self.mp)
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
        x = F.elu(self.fc3(x))
        return x

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
cpu = torch.device('cpu')
net = Net().to(device)


optimizer = optim.Adam(net.parameters(), lr = 0.001)
loss_fn = nn.BCEWithLogitsLoss()

random.shuffle(Data)


THRESH_COUNT = 2
PLT_SHOW = 4
SAVE_MODEL = True
checkpoint_filename = 'Pneumonia50v1.pickle'
random.shuffle(Data)
train_data = Data[1000:]
test_data = Data[:999]



print(f'Running on {device}.')
train_flag = True
save_model_ref = 0
epoch_count = 1
epoch_list = []
val_loss_list = []
acc_list = []
thresh_count = THRESH_COUNT
b = time.time()
while train_flag:
    random.shuffle(train_data)
    train_data = train_data[:1000]
    temp_arr = []
    correct = 0
    total = 0
    print(f'\n----------Training epoch {epoch_count}----------')
    for data in tqdm(train_data):
        X, y = data
        X = X.to(device)
        y = y.to(device)
        
        optimizer.zero_grad()
        out = net(X.float())
        train_loss = loss_fn(out.float(), y.view(-1,N_OUTPUTS).float())
        train_loss.backward()
        optimizer.step()
    print(f'Training loss of epoch {epoch_count} = {train_loss}.\n')
    print(f'----------Testing epoch {epoch_count}----------')
    with torch.no_grad():
        for data in tqdm(test_data):
            X, y = data
            X = X.to(device)
            y = y.to(device)
            out = net(X.float())
            if(out == y.view(-1, N_OUTPUTS)):
                correct += 1
            total += 1
            val_loss = loss_fn(out.float(), y.view(-1,N_OUTPUTS).float())
            temp_arr.append(val_loss.item())
    val_loss_list.append(np.average(temp_arr))
    epoch_list.append(epoch_count)
    acc_list.append(correct/total)
    print(f'Validation accuracy of epoch {epoch_count} = {correct/total*100} Percent.')
    print(f'Validation loss of epoch {epoch_count} = {np.average(temp_arr)}.')
    if epoch_count%PLT_SHOW == 0:
        plot_acc_loss(epoch_list, acc_list, val_loss_list)
    epoch_count += 1
    temp_flag = Train_optimizer(val_loss_list)
    if thresh_count == 0 and temp_flag is False:
        train_flag = False
    if train_loss == 0:
        break
    if temp_flag is False:
        print('****Threshold detected****')
        thresh_count = thresh_count - 1
    else:
        thresh_count = THRESH_COUNT
a = time.time()
print('Training Terminated.')
if SAVE_MODEL is True:
    print(f'Model trained and saved. Reference epoch = {val_loss_list.index(min(val_loss_list))+1}.')
else:
    print(f'Model trained. Reference epoch = {val_loss_list.index(min(val_loss_list))+1}')
print(f'Total time for training = {int((a-b)/3600)} Hrs. {int((a-b)/60)} Min. {round((a-b)%60, 2)} Seconds.')
plot_acc_loss(epoch_list, acc_list, val_loss_list)



