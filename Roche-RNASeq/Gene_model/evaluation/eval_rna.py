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
from pandas.core.algorithms import value_counts

from evaluation.metrics import binary_acc, multi_acc, MCC, MSE, correlation

def evaluate_bin(model, test_loader, device):
    ## Validation
    epoch_loss = 0
    epoch_acc = 0
    y_pred_list = []
    model.eval()
    # Iterate over data.
    for X_batch, y_batch in test_loader:
      with torch.no_grad():
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        y_test_pred = model(X_batch)
        criterion = nn.BCEWithLogitsLoss()
        loss = criterion(y_test_pred, y_batch.unsqueeze(1))
        acc = binary_acc(y_test_pred, y_batch.unsqueeze(1))     
        y_test_pred = torch.sigmoid(y_test_pred)
        y_pred_tag = torch.round(y_test_pred)
      epoch_loss += loss.item()
      epoch_acc += acc.item()
    loss_mean = epoch_loss/len(test_loader)
    acc_mean = epoch_acc/len(test_loader)
    return loss_mean, acc_mean


def evaluate_multi(model, test_loader, device):
    ## Validation
    epoch_loss = 0
    epoch_acc = 0
    y_pred_list = []
    model.eval()
    # Iterate over data.
    for X_batch, y_batch in test_loader:
      with torch.no_grad():
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        y_pred = model(X_batch)
        criterion = CrossEntropyLoss()
        loss = criterion(y_pred, y_batch)
        acc = multi_acc(y_pred, y_batch)
      epoch_loss += loss.item()
      epoch_acc += acc.item()
    loss_mean = epoch_loss/len(test_loader)
    acc_mean = epoch_acc/len(test_loader)
    return loss_mean, acc_mean


def evaluate_regr(model, test_loader, device):
    ## Validation
    epoch_loss = 0
    epoch_acc = 0
    y_pred_list = []
    model.eval()
    # Iterate over data.
    for X_batch, y_batch in test_loader:
      with torch.no_grad():
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        y_pred = model(X_batch)
        criterion = MSELoss()
        loss = criterion(y_pred, y_batch.unsqueeze(1))
        acc = MSE(y_pred, y_batch.unsqueeze(1))
      epoch_loss += loss.item()
      epoch_acc += acc.item()
    loss_mean = epoch_loss/len(test_loader)
    acc_mean = epoch_acc/len(test_loader)
    return loss_mean, acc_mean
