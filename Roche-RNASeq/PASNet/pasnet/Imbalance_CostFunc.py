import torch
import torch.nn.functional as F


def multi_class_entropy(predict, target, class_symbol = 0):
	'''calculate cross entropy in average for samples who belong to the same class.
	lts = False: non-LTS samples are obtained.
	'''
	lts_idx = torch.argmax(target, dim = 1)
	y = target[lts_idx == class_symbol]
	pred = predict[lts_idx == class_symbol]
	cost = F.binary_cross_entropy(pred, y)
	return(cost)

def cross_entropy_for_imbalance(predict, target, class_num):
	'''claculate cross entropy for imbalance data in binary classification.'''
	total_cost = 0 
	for i in range(class_num):
		total_cost = total_cost + multi_class_entropy(predict, target, class_symbol = i)
	return(total_cost)
	

# def bce_for_one_class(predict, target, lts = False):
# 	'''calculate cross entropy in average for samples who belong to the same class.
# 	lts = False: non-LTS samples are obtained.
# 	'''
# 	lts_idx = torch.argmax(target, dim = 1)
# 	if lts == False:
# 		idx = 0 # label = 0, non-LTS
# 	else: idx = 1 # label = 1, LTS
# 	y = target[lts_idx == idx]
# 	pred = predict[lts_idx == idx]
# 	cost = F.binary_cross_entropy(pred, y)

# 	return(cost)

# def binary_cross_entropy_for_imbalance(predict, target):
# 	'''claculate cross entropy for imbalance data in binary classification.'''
# 	total_cost = bce_for_one_class(predict, target, lts = True) + bce_for_one_class(predict, target, lts = False)

# 	return(total_cost)
	
