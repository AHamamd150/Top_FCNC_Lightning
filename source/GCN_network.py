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
from .Graph_networks import GCN,EdgeConv,GAT
import torch_geometric
#####################################################################################
#####################################################################################
#####################################################################################
#####################################################################################
#####################################################################################
class GCNModel(LightningModule):
    def __init__(self,
            In_feat: int=10,
            H_feat: int=8,
            Num_classes: int=1,
            num_layers: int=5,
            num_heads: int= 4,
            dropout: float=0.2,
            lr: float= 0.001)-> None:
        super().__init__()
        self.In_feat = In_feat
        self.H_feat = H_feat
        self.num_layers= num_layers
        self.num_heads= num_heads
        self.dropout= dropout
        self.Num_classes = Num_classes
        self.lr = lr
          
          
        self.model = GAT(in_feat=self.In_feat,
                      h_feat=self.H_feat,
                      n_layers= self.num_layers,
                      n_heads = self.num_heads,
                      dropout = self.dropout,
                      num_classes=Num_classes) 
        
        #self.model = EdgeConv(in_feat=self.In_feat,
        #              h_feat=self.H_feat,
        #              num_layers = self.num_layers,
        #              num_classes = self.Num_classes)
        self.accuracy = Accuracy(task="binary")
        self.CM = BinaryConfusionMatrix()
        self.AUC = BinaryAUROC(thresholds=None)
       

                      
    def forward(self,x):
        return self.model(x)       

             
    def training_step(self, batch, batch_idx):
                        
        x, y = batch
            
        y_hat = self(x).flatten()       
        loss = torch.nn.functional.binary_cross_entropy(y_hat, y)
        acc = self.accuracy(y_hat, y)
        self.log("loss", loss, prog_bar=True)
        self.log("Acc", acc, prog_bar=True)
        return loss

          
    def validation_step(self, batch, batch_idx):
        x,y = batch
        y_hat = self(x).flatten()       
        loss = torch.nn.functional.binary_cross_entropy(y_hat, y)
        acc = self.accuracy(y_hat, y)
        auc = self.AUC(y_hat, y)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_Acc", acc, prog_bar=True)
        self.log("val_AUC", auc, prog_bar=True)
        return loss
  
    def test_step(self, batch, batch_idx):
                    
        x,y = batch
        y_hat = self(x).flatten()       
        loss = torch.nn.functional.binary_cross_entropy(y_hat, y)
        test_bcm = self.CM(y_hat, y)
        test_acc = self.accuracy(y_hat,y)

        test_auc = self.AUC(y_hat,y)
        FP, TP, _ = self.roc_curve(y_hat,torch.tensor(y,dtype = torch.long))

        self.log("loss", loss, prog_bar=True)
        self.log("test_loss", loss, prog_bar=True, sync_dist=True)
        self.log("test_acc", test_acc, prog_bar=True, sync_dist=True)
        self.log("test_auc", test_auc, prog_bar=True, sync_dist=True)
        #self.log("test_TP", TP, prog_bar=True, sync_dist=True)
        #self.log("test_TN", FP, prog_bar=True, sync_dist=True)
        torch.save(TP,'tpr.pt')
        torch.save(FP,'fpr.pt')
        return loss, test_acc,test_auc


    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-2)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=1e-5 )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}



