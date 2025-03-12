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
from .Transformer_Encoder import TransformerEncoder
#####################################################################################
#####################################################################################
#####################################################################################
#####################################################################################
#####################################################################################
class TransformerModel(LightningModule):
    def __init__(self, 
            input_dim: int=10, 
            embed_dim: list=[512,256,128],
            h_dim: int=128,
            num_layers: int=10, 
            expansion_factor: int=4,
            n_heads: int=8,
            masked: bool=True,
            lr: float= 0.001)-> None:
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.h_dim = h_dim
        self.num_layers = num_layers
        self.expansion_factor = expansion_factor
        self.n_heads = n_heads
        self.masked = masked
        self.lr = lr
        
       
        self.model = TransformerEncoder(input_dim=self.input_dim,
        embed_dim=self.embed_dim, 
        h_dim=self.h_dim,
        num_layers=self.num_layers, 
        expansion_factor=self.expansion_factor,
        n_heads=self.n_heads,
        masked=self.masked)
        self.accuracy = Accuracy(task="binary")
        self.CM = BinaryConfusionMatrix()
        self.AUC = BinaryAUROC(thresholds=None)


    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x).flatten()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(y_hat, y)
        acc = self.accuracy(y_hat, y)
        self.log("loss", loss, prog_bar=True)
        self.log("Acc", acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x).flatten()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(y_hat, y)
        acc = self.accuracy(y_hat, y)
        auc = self.AUC(y_hat, y)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        self.log("val_AUC", auc, prog_bar=True)
        return loss
        
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x).flatten()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(y_hat, y)
        self.log("val_loss", loss, prog_bar=True)
        return loss   
        

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """
        Configure the optimizer and learning rate scheduler.

        Returns:
            dict: A dictionary containing the optimizer and learning rate scheduler.
        """
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-2)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=20, eta_min=1e-5
        )
       
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


