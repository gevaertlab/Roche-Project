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
import pickle
import matplotlib.pyplot as plt



def binary_acc(y_pred, y_test):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))

    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum/y_test.shape[0]
    acc = torch.round(acc * 100)

    return acc

def multi_acc(y_pred, y_test):
    y_pred_softmax = torch.log_softmax(y_pred, dim = 1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim = 1)

    correct_pred = (y_pred_tags == y_test).float()
    acc = correct_pred.sum() / len(correct_pred)

    acc = torch.round(acc * 100)

    return acc


# Function to calculate the Matthews correlation coefficient
def MCC(logits, labels):
    pred_flat = np.argmax(logits, axis=1).flatten()
    labels_flat = labels.flatten()
    atthews = matthews_corrcoef(pred_flat, labels_flat)
    return atthews

# Function to calculate the mean squred arror of our predictions vs labels
# The input are predicted values and labels from each batch

def MSE(y_pred, y_true):
  y_pred = y_pred.detach().cpu().numpy()
  y_true = y_true.detach().cpu().numpy()
  return mean_squared_error(y_pred, y_true)

# Function to calculate R^2
def correlation(y_pred, y_true):
  y_pred = y_pred.detach().cpu().numpy().flatten()
  y_true = y_true.detach().cpu().numpy().flatten()
  r= np.corrcoef(y_pred,y_true)[0,1]
  #print(r)
  return r**2
