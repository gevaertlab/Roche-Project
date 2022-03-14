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
from pandas.core.algorithms import value_counts
import matplotlib.pyplot as plt

from model.models import MultiClassification, Regression, PathwayClassification
from dataloader.dataloader_rna import TrainData, ValData, TestData
from evaluation.eval_rna import evaluate_multi, evaluate_regr
from evaluation.metrics import binary_acc, multi_acc, MCC, MSE, correlation
from train.train_multi import train_multi
from train.train_regr import train_regr
from train.predict import predict_multi, predict_regr

''' Initiate configuration '''
parser = argparse.ArgumentParser(description='SSL training')
parser.add_argument('--config', type=str, help='JSON config file')
parser.add_argument('--seed', type=int, default=69,
        help='Seed for random generation')
parser.add_argument('--flag', type=str, default=None,
        help='Flag to use for saving the checkpoints')

args = parser.parse_args()
with open(args.config) as f:
    config = json.load(f)

print(10*'-')
print('Config for this experiment \n')
print(config)
print(10*'-')

print(10*'-')
print('Args for this experiment \n')
print(args)
print(10*'-')

if not args.flag:
    args.flag = 'train_{date:%Y-%m-%d %H:%M:%S}'.format(date=datetime.datetime.now())

np.random.seed(args.seed)
torch.manual_seed(args.seed)

data_path = config[ "csv_path"]
col_name = config["col_name"]
batch_size = config["batch_size"]
num_epoches = config["num_epoches"]
cancer = config["cancer"]

data = pd.read_csv(data_path, index_col = 0, dtype = {'gender': 'category'})
data = data[data[col_name].notnull()].copy()
data. sort_values(by=[col_name], inplace = True)
print("data shape:", data.shape)

# convert labels to class ids
class2idx = {}
i = 0
for t in data[col_name].unique():
  class2idx[t] = i
  i = i + 1
idx2class = {v: k for k, v in class2idx.items()}
data[col_name].replace(class2idx, inplace=True)
print(class2idx)

# train test split
X = data.filter(regex = ("rna*"))
y = data.loc[:, col_name]
X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.2, stratify = y, random_state=args.seed)
X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval,test_size=0.2, stratify=y_trainval, random_state=args.seed)
print(X_train.shape, X_val.shape, X_test.shape)

# scale the dataset
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# load the data
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

train_loader = DataLoader(dataset=train_data, batch_size = batch_size, sampler = weighted_sampler)
val_loader = DataLoader(dataset=val_data, batch_size=1)
test_loader = DataLoader(dataset=test_data, batch_size=1)

summary_writter = os.path.join(config['summary_dir'],
                            datetime.datetime.now().strftime("%Y-%m-%d") + "_" + cancer + "_{0}".format("oncogenic_cla") + ".txt")
file_name = "model_" + cancer + "_oncogenic_cla.pt"

# train the model
device = torch.device("cuda" if (torch.cuda.is_available() and config['use_cuda'] == 0) else "cpu")
print(device)
num_features = X.shape[1]
num_labels = len(class2idx)
model =  MultiClassification(num_features = num_features, num_labels = num_labels)
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=config["lr_rna"])

acc, loss = train_multi(model, train_loader, val_loader, num_labels, optimizer, device,
          class_weights = class_weights,
          num_epochs = num_epoches, save_dir = config['save_dir'], file_name = file_name,summary_writter = summary_writter)

# predict the stage using optimized model
model_path = os.path.join(config['save_dir'], file_name)
model.load_state_dict(torch.load(model_path))
y_pred = predict_multi(model, test_loader, device)
print(y_pred)

confusion_matrix_df = pd.DataFrame(confusion_matrix(y_test, y_pred, normalize = "true")).rename(columns=idx2class, index=idx2class)
print(confusion_matrix_df)
plt.figure()
g = sns.heatmap(confusion_matrix_df, annot=True)
g.set(title=cancer)
g.set_xlabel("Predicted", fontsize = 20)
g.set_ylabel("Actual", fontsize = 20)

idx2class_str = {str(v): k for k, v in class2idx.items()}
report = classification_report(y_test, y_pred, output_dict=True)
df_report = pd.DataFrame(report).transpose().rename(index = idx2class_str)
print(f1_score(y_test, y_pred, average= "weighted"))
df_report