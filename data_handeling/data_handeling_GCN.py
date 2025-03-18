import torch
from torch.utils.data import Dataset
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
import scipy
import sklearn.preprocessing as sklp
import networkx as  nx
import torch_geometric
import torch_geometric.data as PyGData
from torch_geometric.utils import to_networkx
from torch_geometric.loader import DataLoader 
import torch_geometric.utils as pyg_utils
from torch_geometric.utils import dense_to_sparse
##############################################
##############################################
##############################################
class GraphDataset(torch_geometric.data.InMemoryDataset):
    """
    PyTorch class to generate graph data
    """
    def __init__(self,
                 data, 
                 labels):
        # read in the ascii file
        self.labels = labels

        self.graphs = data

    def __len__(self):
        return len(self.graphs)  

    def __getitem__(self, idx):
        graph = self.graphs[idx]
        n_nodes = graph.shape[0]
        # edge index (use complete graph without edge weighting)
        
        adj_matrix = torch.ones((n_nodes, n_nodes)) - torch.eye(n_nodes)
        edges, _ = dense_to_sparse(adj_matrix)
        # node features
        x = graph #graph[['I1', 'I2', 'I3', 'I4', 'I5',  'Pt','Eta', 'Phi','E', 'Charge']].to_numpy()
        d1 = x[edges[0], 6]-x[edges[1], 6]
        d2 = x[edges[0], 7]-x[edges[1], 7]
        dist = np.hypot(d1,d2)
        dist = torch.tensor(dist, dtype=torch.float32)
        
        # edge index and edge attributes
        # compute pair-wise distance matrix from (eta, phi)
        y = self.labels[idx]
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)
        g = PyGData.Data(x=x, edge_index=edges, edge_attr=dist)

       
        return  g, y
#####################################################
#####################################################
#####################################################
#####################################################
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
        #sig = pd.read_csv(self.dir_+'topGamma/Variables-Transformers.csv',delimiter=',',header=None)
        sig = pd.read_csv(self.dir_+'/ttbarToGammaUp/Variables-Transformers.csv',delimiter=',',header=None)
        #sig = pd.read_csv(self.dir_+'/ttbarToGammaCharm/Variables-Transformers.csv',delimiter=',',header=None)
        
        w1 = 7.025e-7
        w2 = 1.12e-3
        w3 = 5.17e-5
        w4 = 4.677e-5
        w5 = 7.1e-6
        l_sig = round(self.n_event/2)

        w = w1+w2+w3+w4+w5
        w1_frac = round((w1/w)*l_sig)
        w2_frac = round((w2/w)*l_sig)
        w3_frac = round((w3/w)*l_sig)
        w4_frac = round((w4/w)*l_sig)
        w5_frac = round((w5/w)*l_sig)
        #print(bkg1.values[...,:-1].reshape(round(bkg1.shape[0]/9),9,10).shape)
        #print(bkg2.values[...,:-1].reshape(round(bkg2.shape[0]/9),9,10).shape)
        #print(bkg3.values[...,:-1].reshape(round(bkg3.shape[0]/9),9,10).shape)
        #print(bkg4.values[...,:-1].reshape(round(bkg4.shape[0]/9),9,10).shape)
        #print(bkg5.values[...,:-1].reshape(round(bkg5.shape[0]/9),9,10).shape)
         
        ######## Reshape ###########
        bkg1 = bkg1.values[...,:-1].reshape(round(bkg1.shape[0]/9),9,10)[:w1_frac,...]
        bkg2 = bkg2.values[...,:-1].reshape(round(bkg2.shape[0]/9),9,10)[:w2_frac,...]
        bkg3 = bkg3.values[...,:-1].reshape(round(bkg3.shape[0]/9),9,10)[:w3_frac,...]
        bkg4 = bkg4.values[...,:-1].reshape(round(bkg4.shape[0]/9),9,10)[:w4_frac,...]
        bkg5 = bkg5.values[...,:-1].reshape(round(bkg5.shape[0]/9),9,10)[:w5_frac,...]
        sig = sig.values[...,:-1].reshape(round(sig.shape[0]/9),9,10)[:l_sig,...]
        background = np.concatenate((bkg1,bkg2,bkg3,bkg4,bkg5),axis=0)
        #print(bkg1.shape,bkg2.shape,bkg3.shape,bkg4.shape,bkg5.shape,)
        print(f'Signal Events:  {sig.shape},   Background Events:   {background.shape}')
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

        self.train_dataset = GraphDataset(self.x_train,self.y_train)
        self.val_dataset =   GraphDataset(self.x_val,self.y_val)
        self.test_dataset =  GraphDataset(self.x_test,self.y_test)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)
#######################################
#######################################
#######################################





