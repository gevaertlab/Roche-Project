import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from sklearn.metrics import roc_auc_score, f1_score
from EvalFunc import balanced_acc
from Imbalance_CostFunc import cross_entropy_for_imbalance
from Model import MultiClassification

def trainGene(train_x, train_y, eval_x, eval_y, \
              In_Nodes,Out_Nodes,\
              Learning_Rate, L2_Lambda, nEpochs, \
              Dropout_Rates = 0.5, optimizer = "Adam", \
              train_writter = None):
    
    net = MultiClassification(num_features = In_Nodes, num_labels = Out_Nodes, Dropout_Rates = Dropout_Rates)

     ###if gpu is being used
    if torch.cuda.is_available():
        net.cuda()
	###

	###the default optimizer is Adam
    if optimizer == "SGD":
        opt = optim.SGD(net.parameters(), lr=Learning_Rate, weight_decay = L2_Lambda)
    else: opt = optim.Adam(net.parameters(), lr=Learning_Rate, weight_decay = L2_Lambda)

    for epoch in range(1, nEpochs + 1):
        print('Epoch {}/{}'.format(epoch, nEpochs))
        print('-' * 10)
        ## TRAIN
        net.train()
        opt.zero_grad()

        pred = net(train_x)
        loss = cross_entropy_for_imbalance(pred, train_y, Out_Nodes) ###calculate train loss
        acc = balanced_acc(pred, train_y) ###calculate train accuracy
        
        output = f'Epoch {epoch+0:03}: | train_loss: {loss:.5f} | train_acc: {acc:.3f}\n'
        if train_writter is not None:
            with open(train_writter, "a+") as f:
                f.write(output)

        loss.backward() ###calculate gradients
        opt.step() ###update weights and biases

        if epoch == nEpochs - 1:
            net.train()
            train_pred = net(train_x)
            train_loss = cross_entropy_for_imbalance(train_pred, train_y, Out_Nodes).view(1,)

            net.eval()
            eval_pred = net(eval_x)
            eval_loss = cross_entropy_for_imbalance(eval_pred, eval_y, Out_Nodes).view(1,)
        
    return (train_pred, eval_pred, train_loss, eval_loss, net)