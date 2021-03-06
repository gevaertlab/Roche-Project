import os
import json
import argparse
import datetime

from DataLoader import load_data
from Train_Gene import trainGene
from EvalFunc import auc, f1, balanced_acc, generate_confusion_matrix
from visulization import view_class
import pandas as pd
import torch
import numpy as np
import pdb 
from preproc import generate_Kfold_index, read_data_gene


''' Initiate configuration '''
parser = argparse.ArgumentParser(description='SSL training')
parser.add_argument('--config', type=str, help='JSON config file')
parser.add_argument('--seed', type=int, default=99,
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

rna_path = config["rna_path"]
col_name = config["col_name"]
selected_class = config["selected_class"]
output_dir = config["output_dir"]
epoch = config["epoch"]
test_size = config["test_size"]
K_fold = config["K_fold"]
Dropout_Rates = config["Dropout_Rates"]
dtype = torch.FloatTensor

if not os.path.exists(output_dir): os.mkdir(output_dir)

''' load data and converted class to labels '''
data, class2idx, idx2class = read_data_gene(rna_path, col_name, selected_class)
plt = view_class(data, col_name, idx2class, output_dir, "class_distribution.png")
print("data shape", data.shape)
print("class2idx", class2idx)

''' Net Settings'''
In_Nodes = data.shape[1] - 1
Out_Nodes = len(data[col_name].unique())

print("Nodes number in each layer:", In_Nodes, Out_Nodes)

''' Initialize '''
nEpochs = epoch ###for training
opt_lr = 1e-4
opt_l2 = 3e-4
if not os.path.exists(output_dir): os.mkdir(output_dir)
train_writter = os.path.join(output_dir, "train_"+ '{date:%Y-%m-%d %H:%M:%S}'.format(date=datetime.datetime.now()) + ".txt")
summary_writter = os.path.join(output_dir, "summary_" + '{date:%Y-%m-%d %H:%M:%S}'.format(date=datetime.datetime.now()) + ".txt")
f= open(summary_writter,"w+")
f0= open(train_writter,"w+")

# split datasets into K folds and get the index list of each fold
train_index_list, test_index_list = generate_Kfold_index(K_fold, data, col_name, test_size)
for fold in range(K_fold):
	print("fold: ", fold)
	x_train, y_train, x_test, y_test = load_data(data, col_name, train_index_list[fold], test_index_list[fold], dtype)
	print("train size:" + str(len(train_index_list[fold])))
	print("test size:" + str(len(test_index_list[fold])))
	pred_train, pred_test, loss_train, loss_test, model = trainGene(x_train, y_train, x_test, y_test, \
														In_Nodes, Out_Nodes, \
														opt_lr, opt_l2, nEpochs, Dropout_Rates, optimizer = "Adam", \
														train_writter = train_writter)
	# evaluate model performence and save the results
	auc_te = auc(y_test, pred_test)
	f1_te = f1(y_test, pred_test)
	acc = balanced_acc(y_test, pred_test)
	output = f'Fold: {fold}: | AUC: {auc_te:.5f} | F1: {f1_te:.3f} | Accuracy:{acc:.5f} \n'
	print(output)
	if summary_writter is not None:
		with open(summary_writter ,"a+") as f:
			f.write(output)
	### Save model  and confusion matrix
	save_model = "model_fold_"+str(fold)+ '_{date:%Y-%m-%d %H:%M:%S}'.format(date=datetime.datetime.now()) + ".pt" # file name to save the model
	save_pic =  "confusion_matrix_fold_"+str(fold)+'_{date:%Y-%m-%d %H:%M:%S}'.format(date=datetime.datetime.now()) + ".png" # file name to save the confustion matrix
	if(output_dir is not None):
		torch.save(model.state_dict(), os.path.join(output_dir, save_model))
		generate_confusion_matrix(pred_test, y_test, idx2class, output_dir, save_pic)

