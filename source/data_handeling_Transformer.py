import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pytorch_lightning as pl
from lightning.pytorch import LightningModule, LightningDataModule
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from torchmetrics import Accuracy
from torchmetrics.classification import BinaryConfusionMatrix, BinaryAUROC
##################
class CustomDataset(Dataset):
    def __init__(self, data, labels):
       
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)  # For classification

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

##############################################
##############################################
##############################################
class CustomDataModule(LightningDataModule):
    def __init__(self,dir_,n_event, batch_size=32):
        super().__init__()
        self.dir_ = dir_  
        self.n_event = n_event
        self.batch_size = batch_size

        ##### Load the data points ######
        bkg1 = pd.read_csv(self.dir_+'/tby/Variables-Transformers.csv',delimiter=',',header=None)
        bkg2 = pd.read_csv(self.dir_+'/WyPlusJets/Variables-Transformers.csv',delimiter=',',header=None)
        bkg3 = pd.read_csv(self.dir_+'/tjy/Variables-Transformers.csv',delimiter=',',header=None)
        bkg4 = pd.read_csv(self.dir_+'/tty/Variables-Transformers.csv',delimiter=',',header=None)
        bkg5 = pd.read_csv(self.dir_+'/twy/Variables-Transformers.csv',delimiter=',',header=None)
        sig = pd.read_csv(self.dir_+'topGamma/Variables-Transformers.csv',delimiter=',',header=None)

        ######## Reshape ###########
        bkg1 = bkg1.values[...,:-1].reshape(round(bkg1.shape[0]/9),9,10)[:self.n_event]
        bkg2 = bkg2.values[...,:-1].reshape(round(bkg2.shape[0]/9),9,10)[:self.n_event]
        bkg3 = bkg3.values[...,:-1].reshape(round(bkg3.shape[0]/9),9,10)[:self.n_event]
        bkg4 = bkg4.values[...,:-1].reshape(round(bkg4.shape[0]/9),9,10)[:self.n_event]
        bkg5 = bkg5.values[...,:-1].reshape(round(bkg5.shape[0]/9),9,10)[:self.n_event]
        sig = sig.values[...,:-1].reshape(round(sig.shape[0]/9),9,10)[:self.n_event*5]
        background = np.concatenate((bkg1,bkg2,bkg3,bkg4,bkg5),axis=0)
        ### Create labels, shuffle, etc ### 
        labels = [0]*len(background)+[1]*len(sig)
        data_ = np.concatenate((background,sig))
        data_, labels = shuffle(data_,labels)
        x_train1,self.x_test,y_train1,self.y_test = train_test_split(data_,np.array(labels),shuffle=True,test_size=0.25)
        self.x_train,self.x_val,self.y_train,self.y_val = train_test_split(x_train1,y_train1,shuffle=True,test_size=0.20)
        print('#==================================================#')
        print(f'Shape of the training dataset: {self.x_train.shape}')
        print('#==================================================#')
        print(f'Shape of the validation dataset: {self.x_val.shape} ')
        print('#==================================================#')
        print(f'Shape of the test dataset: {self.x_test.shape} ')
        print('#==================================================#')
    def setup(self, stage=None):

        self.train_dataset = CustomDataset(self.x_train, self.y_train)
        self.val_dataset = CustomDataset(self.x_val, self.y_val)
        self.test_dataset = CustomDataset(self.x_test, self.y_test)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)
