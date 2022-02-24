from audioop import reverse
from DataLoader import load_data, load_pathway
from Train import trainPASNet
from Model import PASNet
from EvalFunc import auc, f1, multi_acc
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import pdb

import torch
import numpy as np

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
''' load data and pathway '''
data = pd.read_csv("data/rna_data_GO.csv")
pathway_mask = load_pathway("data/pathway_mask_GO.csv", dtype)

model = PASNet(In_Nodes, Pathway_Nodes, Hidden_Nodes, Out_Nodes, pathway_mask)
model_path = "Results/GO/model_fold_3.pt"
model.load_state_dict(torch.load(model_path))

df_pathway = pd.read_csv("data/pathway_mask_GO.csv", index_col = 0)
pathway_nodes = df_pathway.index.values.tolist()
gene_nodes = df_pathway.columns.values.tolist()
print(model)
gene_pathway_weights = model.sc1.weight.data.numpy()
pathway_hidden_weights = model.sc2.weight.data.numpy()
hidden_outcome_weights = model.sc3.weight.data.numpy()
print(pathway_hidden_weights.shape)
abs_weights_hidden = np.abs(pathway_hidden_weights).sum(axis = 0)
ranked_pathways = [x for _, x in sorted(zip(abs_weights_hidden, pathway_nodes), reverse=True)]
ranked_weights = sorted(abs_weights_hidden, reverse = True)
df_top_pathways = pd.DataFrame({"Pathway": ranked_pathways, "Weight": ranked_weights})
df_top_pathways = df_top_pathways.iloc[0:20,]
#print(df_top_pathways)

# Find top genes in the top pathways
Top20Path = ranked_pathways[0:20]
top_gene_list = []
for path in Top20Path:
    idx_path = pathway_nodes.index(path)
    abs_gene_weights = np.abs(gene_pathway_weights[idx_path])
    ranked_genes = [x for _, x in sorted(zip(abs_gene_weights, gene_nodes), reverse=True)]
    top_genes = ",".join(ranked_genes[0:5])
    top_gene_list.append(top_genes)
df_top_pathways['Genes'] = top_gene_list

df_top_pathways.to_csv("Results/GO/Top_GO.csv")



