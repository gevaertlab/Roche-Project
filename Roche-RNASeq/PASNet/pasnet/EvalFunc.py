import torch

from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix, balanced_accuracy_score
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def auc(y_true, y_pred):
	###if gpu is being used, transferring back to cpu
	if torch.cuda.is_available():
		y_true = y_true.cpu().detach()
		y_pred = y_pred.cpu().detach()
	###
	auc = roc_auc_score(y_true.detach().numpy(), y_pred.detach().numpy())
	return(auc)

def f1(y_true, y_pred):
	###covert one-hot encoding into integer
	y = torch.argmax(y_true, dim = 1)
	###estimated targets (either 0 or 1)
	pred = torch.argmax(y_pred, dim = 1)
	###if gpu is being used, transferring back to cpu
	if torch.cuda.is_available():
		y = y.cpu().detach()
		pred = pred.cpu().detach()
	###
	f1 = f1_score(y.numpy(), pred.numpy(), average= "weighted")
	return(f1)

def flat_acc(y_pred, y_true):
	###covert one-hot encoding into integer
	y = torch.argmax(y_true, dim = 1)
	###estimated targets (either 0 or 1)
	pred = torch.argmax(y_pred, dim = 1)
	if torch.cuda.is_available():
		y = y.cpu().detach()
		pred = pred.cpu().detach()
	correct_pred = (y == pred).float()
	acc = correct_pred.sum() / len(correct_pred)
	acc = acc.numpy()
	return acc

def balanced_acc(y_pred, y_true):
		###covert one-hot encoding into integer
	y = torch.argmax(y_true, dim = 1)
	###estimated targets (either 0 or 1)
	pred = torch.argmax(y_pred, dim = 1)
	if torch.cuda.is_available():
		y = y.cpu().detach()
		pred = pred.cpu().detach()
	acc = balanced_accuracy_score(y, pred)
	return acc

def generate_confusion_matrix(eval_pred, eval_y, idx2class, save_dir, save_name):
	###covert one-hot encoding into integer
	y = torch.argmax(eval_y, dim = 1)
	pred = torch.argmax(eval_pred, dim = 1)
	if torch.cuda.is_available():
		y = y.cpu().detach()
		pred = pred.cpu().detach()
	confusion_matrix_df = pd.DataFrame(confusion_matrix(y, pred, normalize = "true")).rename(columns=idx2class, index=idx2class)
	#confusion_matrix_df = confusion_matrix_df.div(confusion_matrix_df.sum(axis=1), axis=0)
	plt.figure()
	g = sns.heatmap(confusion_matrix_df, annot=True)
	g.set(title=save_name)
	g.set_xlabel("Predicted", fontsize = 20)
	g.set_ylabel("Actual", fontsize = 20)
	plt.savefig(save_dir + "/" + save_name)


