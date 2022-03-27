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

class BinaryClassification(nn.Module):
    def __init__(self, num_features = 17052):
        super(BinaryClassification, self).__init__()
        self.num_features = num_features
        self.layer_1 = nn.Linear(num_features, 4096) 
        self.layer_2 = nn.Linear(4096, 2048)
        self.layer_3 = nn.Linear(2048, 512) 
        self.layer_4 = nn.Linear(512, 1)  
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        
    def forward(self, inputs):
        x = self.relu(self.layer_1(inputs))
        x = self.dropout(x)
        x = self.relu(self.layer_2(x))
        x = self.dropout(x)
        x = self.relu(self.layer_3(x))
        x = self.dropout(x)
        x = self.layer_4(x)
        return x

class MultiClassification(nn.Module):
    def __init__(self, num_features = 17052, num_labels = 4, Dropout_Rates = 0.5):
        super(MultiClassification, self).__init__()
        self.num_features = num_features
        self.num_labels = num_labels
        self.layer_1 = nn.Linear(num_features, 4096)
        self.layer_2 = nn.Linear(4096, 2048)
        self.layer_3 = nn.Linear(2048, 512)
        self.layer_4 = nn.Linear(512, num_labels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p = Dropout_Rates)

    def forward(self, inputs):
        x = self.relu(self.layer_1(inputs))
        x = self.dropout(x)
        x = self.relu(self.layer_2(x))
        x = self.dropout(x)
        x = self.relu(self.layer_3(x))
        x = self.dropout(x)
        x = self.layer_4(x)
        return x


class PathwayClassification(nn.Module):
    def __init__(self, num_features = 17052, num_labels = 4):
        super(PathwayClassification, self).__init__()
        self.num_features = num_features
        self.num_labels = num_labels
        self.layer_1 = nn.Linear(num_features, 64)
        self.layer_2 = nn.Linear(64, 32)
        self.layer_3 = nn.Linear(32, num_labels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()

    def forward(self, inputs):
        x = self.relu(self.layer_1(inputs))
        x = self.dropout(x)
        x = self.relu(self.layer_2(x))
        x = self.dropout(x)
        x = self.layer_3(x)
        return x


class Regression(nn.Module):
    def __init__(self, num_features = 17052):
        super(Regression, self).__init__()
        self.num_features = num_features
        self.layer_1 = nn.Linear(num_features, 4096)
        self.layer_2 = nn.Linear(4096, 2048)
        self.layer_3 = nn.Linear(2048, 512)
        self.layer_4 = nn.Linear(512, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()

    def forward(self, inputs):
        x = self.relu(self.layer_1(inputs))
        x = self.dropout(x)
        x = self.relu(self.layer_2(x))
        x = self.dropout(x)
        x = self.relu(self.layer_3(x))
        x = self.dropout(x)
        x = self.layer_4(x)
        return x

class PathwayRegression(nn.Module):
    def __init__(self, num_features = 17052):
        super(Regression, self).__init__()
        self.num_features = num_features
        self.layer_1 = nn.Linear(num_features, 64)
        self.layer_2 = nn.Linear(64, 32)
        self.layer_3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()

    def forward(self, inputs):
        x = self.relu(self.layer_1(inputs))
        x = self.dropout(x)
        x = self.relu(self.layer_2(x))
        x = self.dropout(x)
        x = self.layer_3(x)
        return x
