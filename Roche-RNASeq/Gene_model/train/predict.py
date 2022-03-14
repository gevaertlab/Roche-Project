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

def predict_bin(model, test_loader, device):
    y_pred_list = []
    model.eval()
    # Iterate over data.
    for X_batch, _ in test_loader:
      with torch.no_grad():
        X_batch = X_batch.to(device)
        y_test_pred = model(X_batch)
        y_test_pred = torch.sigmoid(y_test_pred)
        y_pred_tag = torch.round(y_test_pred)
        y_pred_list.append(y_pred_tag.detach().cpu().numpy())
    y_pred_list = [a.squeeze().tolist() for a in  y_pred_list]

    return y_pred_list


def predict_multi(model, test_loader, device):
    y_pred_list = []
    model.eval()
    # Iterate over data.
    for X_batch, _ in test_loader:
      with torch.no_grad():
        X_batch = X_batch.to(device)
        y_pred = model(X_batch)
        _, y_pred_tags = torch.max(y_pred, dim = 1)
        y_pred_list.append(y_pred_tags.detach().cpu().numpy())

    y_pred_list = [a.squeeze().tolist() for a in  y_pred_list]
    return y_pred_list

def predict_regr(model, test_loader, device):
    y_pred_list = []
    model.eval()
    # Iterate over data.
    for X_batch, _ in test_loader:
      with torch.no_grad():
        X_batch = X_batch.to(device)
        y_pred = model(X_batch)
        y_pred_list.append(y_pred.detach().cpu().numpy())

    y_pred_list = [a.squeeze().tolist() for a in  y_pred_list]
    return y_pred_list
