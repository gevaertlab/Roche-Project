from DataLoader import load_data, load_pathway
from Train import trainPASNet
from EvalFunc import auc, f1, multi_acc
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import torch
import numpy as np
import pdb 

output_dir = "Results/GO"
dtype = torch.FloatTensor
''' Net Settings'''
#In_Nodes = 3915 ###number of genes in REACTOME
In_Nodes = 13581  ###number of genes in GO
#Pathway_Nodes = 574 ###number of pathways in REACTOME
Pathway_Nodes = 7033 ###number of pathways in GO
Hidden_Nodes = 200 ###number of hidden nodes
Out_Nodes = 3 ###number of hidden nodes in the last hidden layer
''' Initialize '''
nEpochs = 5000 ###for training
Dropout_Rates = [0.8, 0.7] ###sub-network setup
summary_writter = output_dir + "/Summary.txt"
f= open(summary_writter,"w+")
''' load data and pathway '''
data = pd.read_csv("data/rna_data_GO.csv")
pathway_mask = load_pathway("data/pathway_mask_GO.csv", dtype)

K = 5 # number of folds
opt_lr = 1e-4
opt_l2 = 3e-4
# split datasets into K folds
train_index_list = []
test_index_list = []
X = data.drop(["Score"], axis = 1).values
y = data.loc[:, ["Score"]].values
kf = StratifiedKFold(n_splits= K)
for train_index, test_index in kf.split(X, y):
	train_index_list.append(train_index)
	test_index_list.append(test_index)

for fold in range(K):
	print("fold: ", fold)
	save_model = "model_fold_"+str(fold)+".pt" # the name to save the model
	save_pic =  "confusion_matrix_fold_"+str(fold)+ ".png"
	x_train, y_train, x_test, y_test = load_data(data, train_index_list[fold], test_index_list[fold], dtype)
	pred_train, pred_test, loss_train, loss_test = trainPASNet(x_train, y_train, x_test, y_test, pathway_mask, \
														In_Nodes, Pathway_Nodes, Hidden_Nodes, Out_Nodes, \
														opt_lr, opt_l2, nEpochs, Dropout_Rates, optimizer = "Adam", \
														save_dir = output_dir, save_model = save_model, save_pic = save_pic)

	auc_te = auc(y_test, pred_test)
	f1_te = f1(y_test, pred_test)
	acc = multi_acc(y_test, pred_test)
	output = f'Fold: {fold}: | AUC: {auc_te:.5f} | F1: {f1_te:.3f} | Accuracy:{acc:.5f} \n'
	print(output)
	if summary_writter is not None:
		with open(summary_writter ,"a+") as f:
			f.write(output)

