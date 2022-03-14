import torch.nn as nn
from torch.optim import Adam, rmsprop
import torch.optim as optim
import seaborn as sns
import torch
import numpy as np
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import time
import os
import sys
import copy
import json
import argparse
import datetime
from torch.utils.data import Dataset, WeightedRandomSampler, SequentialSampler, RandomSampler, DataLoader, TensorDataset
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, matthews_corrcoef, mean_squared_error
import torch.multiprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, r2_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pickle

from model.models import MultiClassification, Regression
from dataloader.dataloader_rna import TrainData, ValData, TestData
from evaluation.metrics import binary_acc, multi_acc, MCC, MSE, correlation
from evaluation.eval_rna import evaluate_multi, evaluate_regr 

def train_regr(model, train_loader, test_loader, optimizer, device, num_epochs = 10, save_dir = ".", file_name= "model_dict_regr.pt", summary_writter = None):

    if not os.path.isdir(save_dir):
      os.makedirs(save_dir)

    accuracy_stats = {'train': [], "val": []}
    loss_stats = {'train': [],"val": []}

    best_val_loss = np.inf
    for epoch in range(1, num_epochs + 1):
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('-' * 10)

        # Each epoch has a training and validation phase
        ## TRAIN
        model.train()
        epoch_loss = 0
        epoch_acc = 0

        # Iterate over data.
        for X_batch, y_batch in train_loader:
          X_batch, y_batch = X_batch.to(device), y_batch.to(device)
          # zero the parameter gradients
          optimizer.zero_grad()
          y_pred = model(X_batch)
          criterion = MSELoss()
          loss = criterion(y_pred, y_batch.unsqueeze(1))
          acc = MSE(y_pred,y_batch.unsqueeze(1))
          loss.backward()
          optimizer.step()
          epoch_loss += loss.item()
          epoch_acc += acc.item()

        val_loss, val_acc = evaluate_regr(model, test_loader, device)
        loss_stats['train'].append(epoch_loss/len(train_loader))
        loss_stats['val'].append(val_loss)
        accuracy_stats['train'].append(epoch_acc/len(train_loader))
        accuracy_stats['val'].append(val_acc)

        output = f'Epoch {epoch+0:03}: | train_MSE: {epoch_loss/len(train_loader):.5f} | train_MSE: {epoch_acc/len(train_loader):.3f} | eval_MSE:{val_loss:.5f} | eval_MSE:{val_acc:.5f}\n'
        print(output)
        if summary_writter is not None:
          with open(summary_writter ,"a+") as f:
            f.write(output)

        # Save/Update a trained model if the eval_loss is decreased than the saved one
        if(val_loss < best_val_loss):
          best_val_loss = val_loss
          if(len(save_dir)) > 0:
            torch.save(model.state_dict(), os.path.join(save_dir, file_name))

    acc_df = pd.DataFrame.from_dict(accuracy_stats).reset_index().melt(id_vars=['index']).rename(columns={"index":"epochs"})
    loss_df = pd.DataFrame.from_dict(loss_stats).reset_index().melt(id_vars=['index']).rename(columns={"index":"epochs"})

    return acc_df, loss_df
