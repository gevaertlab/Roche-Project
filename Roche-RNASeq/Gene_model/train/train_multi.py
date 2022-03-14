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


def prepare_multi(data, col_name, random_num, summary_writter, file_name, args):
      data = data[data[col_name].notnull()].copy()
      class2idx = {}
      i = 0
      for t in data[col_name].unique():
        class2idx[t] = i
        i = i + 1
      idx2class = {v: k for k, v in class2idx.items()}
      data[col_name].replace(class2idx, inplace=True)
      X = data.filter(regex = ("rna*"))
      y = data.loc[:, col_name]
      X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.2, stratify = y, random_state=random_num)
      X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval,test_size=0.2, stratify=y_trainval, random_state=random_num)
      print(X_train.shape, X_val.shape, X_test.shape)

      scaler = StandardScaler()
      X_train = scaler.fit_transform(X_train)
      X_val = scaler.transform(X_val)
      X_test = scaler.transform(X_test)

      train_data = TrainData(torch.tensor(X_train.astype(np.float32)), torch.tensor(y_train.astype(np.long)))
      val_data = ValData(torch.tensor(X_val.astype(np.float32)), torch.tensor(y_val.astype(np.long)))
      test_data = TestData(torch.tensor(X_test.astype(np.float32)), torch.tensor(y_test.astype(np.long)))

      target_list = []
      for _, t in train_data:
        target_list.append(t)
      target_list = torch.tensor(target_list)

      unique, counts = np.unique(target_list, return_counts=True)
      class_weights = 1./torch.tensor(counts, dtype=torch.float)
      class_weights_all = class_weights[target_list]

      weighted_sampler = WeightedRandomSampler(
        weights=class_weights_all,
        num_samples=len(class_weights_all),
        replacement=True
       )

      train_loader = DataLoader(dataset=train_data, batch_size=args["batch_size"], sampler = weighted_sampler)
      val_loader = DataLoader(dataset=val_data, batch_size=1)
      test_loader = DataLoader(dataset=test_data, batch_size=1)

      device = torch.device("GPU" if (torch.cuda.is_available() and args['use_cuda']) else "cpu")
      print(device)
      num_features = X.shape[1]
      print(num_features)
      num_labels = len(class2idx)
      model =  MultiClassification(num_features = num_features, num_labels = num_labels)
      model.to(device)
      optimizer = optim.Adam(model.parameters(), lr=args["lr_rna"])

      acc, loss = train_multi(model, train_loader, val_loader, num_labels, optimizer, device,
                class_weights = class_weights,
                num_epochs = args['num_epochs'], save_dir = args['save_dir'], file_name = file_name,summary_writter = summary_writter)

      return acc, loss


def train_multi(model, train_loader, test_loader, num_labels, optimizer, device, class_weights = None, num_epochs = 10, save_dir = ".", file_name = "model_dict_multiclass.pt", summary_writter = None):

    if not os.path.isdir(save_dir):
      os.makedirs(save_dir)

    accuracy_stats = {'train': [], "val": []}
    loss_stats = {'train': [],"val": []}

    best_val_loss = np.inf
    for epoch in tqdm(range(1, num_epochs + 1)):
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
          if(len(class_weights)>0):
            criterion = CrossEntropyLoss(weight = class_weights.to(device))
          else:
            criterion = CrossEntropyLoss()
          loss = criterion(y_pred, y_batch)
          acc = multi_acc(y_pred, y_batch)
          loss.backward()
          optimizer.step()
          epoch_loss += loss.item()
          epoch_acc += acc.item()

        val_loss, val_acc  = evaluate_multi(model, test_loader, device)

        loss_stats['train'].append(epoch_loss/len(train_loader))
        loss_stats['val'].append(val_loss)
        accuracy_stats['train'].append(epoch_acc/len(train_loader))
        accuracy_stats['val'].append(val_acc)

        output = f'Epoch {epoch+0:03}: | train_loss: {epoch_loss/len(train_loader):.5f} | train_acc: {epoch_acc/len(train_loader):.3f} | eval_loss:{val_loss:.5f} | eval_acc:{val_acc:.5f}\n'
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
