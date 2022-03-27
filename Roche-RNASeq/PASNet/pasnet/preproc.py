from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import pdb 

def filterGenes(X, pathway):
    # the column names start with "rna_", we need to filter out the prefix and rename the columns
    genes = X.columns.values.tolist()
    genes = [w[4:] for w in genes]
    X.columns = genes
    overlap_genes = list(set(pathway.columns.values.tolist()) & set(genes))
    print("Number of overlap genes:", len(overlap_genes))
    X = X.loc[:, overlap_genes]
    df_pathway = pathway.loc[:, overlap_genes]
    return X, df_pathway

def findCommonGenesPathways(rna, pathway, col_name):
    X = rna.drop(col_name, axis = 1)
    y = rna.loc[:, col_name] 
    overlap_genes = list(set(pathway.columns.values.tolist()) & set(rna.columns.values.tolist()))
    print("Number of overlapping genes:", len(overlap_genes))
    df_rna = rna.loc[:, overlap_genes]
    df_rna[col_name] = y.values
    df_pathway = pathway.loc[:, overlap_genes]
    return df_rna, df_pathway

def read_data(path, col_name, selected_class):
    data = pd.read_csv(path, index_col = 0, low_memory=False)
    data = data.T
    data = data[data[col_name].notnull()].copy()
    if selected_class is not None:
        data = data.loc[data[col_name].isin (selected_class)]
    class2idx = {}
    i = 0
    for t in data[col_name].unique():
        class2idx[t] = i
        i = i + 1
    idx2class = {v: k for k, v in class2idx.items()}
    data[col_name].replace(class2idx, inplace=True)
    return data, class2idx, idx2class

def read_data_gene(path, col_name, selected_class):
    data = pd.read_csv(path, index_col = 0, low_memory=False)
    data = data[data[col_name].notnull()].copy()
    data. sort_values(by=[col_name], inplace = True)
    X = data.filter(regex = ("rna*"))
    y = data.loc[:, col_name]
    X[col_name] = y
    data = X.copy()
    class2idx = {}
    i = 0
    for t in data[col_name].unique():
        class2idx[t] = i
        i = i + 1
    idx2class = {v: k for k, v in class2idx.items()}
    data[col_name].replace(class2idx, inplace=True)
    return data, class2idx, idx2class

def generate_Kfold_index(K, data, col_name, test_size):
    train_index_list = []
    test_index_list = []
    n_samples = data.shape[0]
    indices = np.arange(n_samples)
    X = data.drop([col_name], axis = 1).values
    y = data.loc[:, [col_name]].values

    if K > 1: 
        # do K fold split 
        kf = StratifiedKFold(n_splits= K)
        for train_index, test_index in kf.split(X, y):
            train_index_list.append(train_index)
            test_index_list.append(test_index)
    else:
        (
        data_train,
        data_test,
        labels_train,
        labels_test,
        indices_train,
        indices_test,
        ) = train_test_split(X, y, indices, test_size = test_size, stratify = y)
        train_index_list.append(indices_train)
        test_index_list.append(indices_test)

    return train_index_list, test_index_list




# args = {
#     "random_state": 69
# }

# # Reading in gene and pathway data
# print("Reading in data")
# data = pd.read_csv("../data/PRAD_rna_clinical.csv")
# print("RNA shape:", data.shape)
# pathway = pd.read_csv("../data/pathway_mask_GO.csv", index_col= 0)
# print("Pathway shape:", pathway.shape)


# # Filter NULL values and convert the grade to class ids
# col_name = "primary_gleason_grade"
# data = data[data[col_name].notnull()].copy()
# class2idx = {3:0, 4:1, 5:2}
# idx2class = {v: k for k, v in class2idx.items()}
# data[col_name].replace(class2idx, inplace=True)
# print(class2idx)
# print(data[col_name]. value_counts())

# # Extract the gene and the outcome columns
# X = data.filter(regex = ("rna*"))
# y = data.loc[:, col_name]
# # Find overlap genes between pathway and gene data sets, and reorder columns
# X, df_pathway = filterGenes(X, pathway)
# # Combine X, y values
# X["Score"] = y.values

# print("Data shape:",  X.shape)
# print(X.head())
# print("Pathway shape:", df_pathway.shape)
# print(df_pathway.head())
# X.to_csv("../data/rna_data_GO.csv", index = False)
# df_pathway.to_csv("../data/pathway_mask_GO.csv")



# # Split data into k-fold
# splits = 5
# print("Splitting data into", splits, "folds...")
# kf = KFold(n_splits= splits, random_state=args['random_state'], shuffle=False)
# for train_index, test_index in kf.split(X):
#     X_train, X_test = X[train_index], X[test_index]
#     y_train, y_test = y[train_index], y[test_index]
#     scaler = StandardScaler()
#     X_train = scaler.fit_transform(X_train)
#     X_test = scaler.transform(X_test)



# # train test split
# X = data.filter(regex = ("rna*"))
# y = data.loc[:, col_name]
# genes = X.columns.values.tolist()
# genes = [w[4:] for w in genes]

# X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size = 0.2, stratify = y,  random_state=args['random_state'])
# X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size = 0.2, stratify = y_trainval,  random_state = args['random_state'])

# # scale the dataset
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_val = scaler.transform(X_val)
# X_test = scaler.transform(X_test)

# # save the dataset
# df_train = pd.DataFrame(X_train, columns=genes)
# df_train["Score"] = y_train.tolist()

# df_val  = pd.DataFrame(X_val, columns=genes)
# df_val["Score"] = y_val.tolist()

# df_test = pd.DataFrame(X_test, columns=genes)
# df_test["Score"] = y_test.tolist()

# pathway = pd.read_csv("../data/pathway_mask.csv", index_col= 0)
# print("pathway shape:", pathway.shape)

# overlap_genes = list(set(pathway.columns.values.tolist()) & set(genes))
# print("number of overlap genes:", len(overlap_genes))
# overlap_genes.append("Score")
# df_train = df_train.loc[:, overlap_genes]
# df_val = df_val.loc[:, overlap_genes]
# df_test = df_test.loc[:, overlap_genes]
# df_pathway = pathway.loc[:, overlap_genes[:-1]]

# print(df_train.shape)
# # print(df_train.head())
# print(df_val.shape)
# # print(df_val.head())
# print(df_test.shape)
# # print(df_test.head())
# print(df_pathway.shape)
# #print(df_pathway.head())

# df_train.to_csv("../data/train.csv", index = False)
# df_val.to_csv("../data/validation.csv", index = False)
# df_test.to_csv("../data/test.csv", index = False)
# df_pathway.to_csv("../data/pathway_mask_new.csv")






