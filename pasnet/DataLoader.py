import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

def vectorized_label(target, n_class):
	'''convert target(y) to be one-hot encoding format(dummy variable)
	'''
	TARGET = np.array(target).reshape(-1)

	return np.eye(n_class)[TARGET]


def convert_data(data, dtype, scaler = None):
	'''Load data, and then covert it to a Pytorch tensor.
	Input:
		data: train or test dataset from the k fold
		dtype: define the data type of tensor (i.e. dtype=torch.FloatTensor)
	Output:
		X: a Pytorch tensor of 'x'.
		Y: a Pytorch tensor of 'y'(one-hot encoding).
	'''	
	x = data.drop(["Score"], axis = 1).values
	y = data.loc[:, ["Score"]].values

	if scaler is None: #train data, initiate scaler and fit data
		scaler = StandardScaler()
		x = scaler.fit_transform(x)
	else:  # test data, just tranform the data
		x = scaler.transform(x)

	X = torch.from_numpy(x).type(dtype)
	Y = torch.from_numpy(vectorized_label(y, 3)).type(dtype)
	###if gpu is being used
	if torch.cuda.is_available():
		X = X.cuda()
		Y = Y.cuda()
	###
	return(X, Y, scaler)

def load_data(data, train_index, test_index, dtype):
	'''Load data, and then covert it to a Pytorch tensor.
	Input:
		data: dataframe of the entire datasets.
		train_index: row ids used for train 
		test_index: row ids used for test
		dtype: define the data type of tensor (i.e. dtype=torch.FloatTensor)
	Output:
		X: a Pytorch tensor of 'x'.
		Y: a Pytorch tensor of 'y'(one-hot encoding).
	'''
	df_train = data.iloc[train_index, :]
	df_test = data.iloc[test_index, :]
	X_train, y_train, scaler = convert_data(df_train, dtype, scaler = None)
	X_test, y_test, _ = convert_data(df_test, dtype, scaler)

	# X_train  = df_train.drop(["Score"], axis = 1).values
	# y_train = df_train.loc[:, ["Score"]].values
	# X_test = df_test.drop(["Score"], axis = 1).values
	# y_test = df_test.loc[:, ["Score"]].values
	# scaler = StandardScaler()
	# X_train = scaler.fit_transform(X_train)
	# X_test = scaler.transform(X_test)
	# X_train = torch.from_numpy(X_train).type(dtype)
	# y_train = torch.from_numpy(vectorized_label(y_train, 3)).type(dtype)
	# X_test = torch.from_numpy(X_test).type(dtype)
	# y_test = torch.from_numpy(vectorized_label(y_test, 3)).type(dtype)
	# ###if gpu is being used
	# if torch.cuda.is_available():
	# 	X_train = X_train.cuda()
	# 	y_train = y_train.cuda()
	# 	X_test = X_test.cuda()
	# 	y_test = y_test.cuda()
	# ###
	return(X_train, y_train, X_test , y_test)


def load_pathway(path, dtype):
	'''Load a bi-adjacency matrix of pathways, and then covert it to a Pytorch tensor.
	Input:
		path: path to input dataset (which is expected to be a csv file).
		dtype: define the data type of tensor (i.e. dtype=torch.FloatTensor)
	Output:
		PATHWAY_MASK: a Pytorch tensor of the bi-adjacency matrix of pathways.
	'''
	pathway_mask = pd.read_csv(path, index_col = 0).as_matrix()

	PATHWAY_MASK = torch.from_numpy(pathway_mask).type(dtype)
	###if gpu is being used
	if torch.cuda.is_available():
		PATHWAY_MASK = PATHWAY_MASK.cuda()
	###
	return(PATHWAY_MASK)
