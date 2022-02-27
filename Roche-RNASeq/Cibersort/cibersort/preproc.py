import pandas as pd
import pdb 

def filterGenes(X, pathway):
    # the column names start with "rna_", we need to filter out the prefix and rename the columns
    genes = X.columns.values.tolist()
    genes = [w[4:] for w in genes]
    X.columns = genes
    overlap_genes = list(set(pathway.index.tolist()) & set(genes))
    print("Number of overlap genes:", len(overlap_genes))
    X = X.loc[:, overlap_genes]
    return X

# Reading in gene and pathway data
print("Reading in data")
data = pd.read_csv("../PASNet-prostate/data/PRAD_rna_clinical.csv", index_col = 0)
print("RNA shape:", data.shape)
pathway = pd.read_csv("data/LM22.txt", sep="\t", index_col = 0)
print("Pathway shape:", pathway.shape)

# Filter NULL values and convert the grade to class ids
col_name = "primary_gleason_grade"
data = data[data[col_name].notnull()].copy()
print(data[col_name]. value_counts())

# Extract the gene and the outcome columns
X = data.filter(regex = ("rna*"))
y = data.loc[:, col_name]
# Find overlap genes between pathway and gene data sets, and reorder columns
X = filterGenes(X, pathway)
# Combine X, y values
#X["Score"] = y.values
X.index.name = "Gene"
X = X.T

X.to_csv("data/rna.txt", sep = "\t")

#df_3 = X[X.Score==3].T.drop("Score")
#df_4 = X[X.Score==4].T.drop("Score")
#df_5 = X[X.Score==5].T.drop("Score")

#print("Data shape:",  df_3.shape)
#print("Data shape:",  df_4.shape)
#print("Data shape:",  df_5.shape)

#df_3.to_csv("../data/rna_pattern_3.txt", sep = "\t")
#df_4.to_csv("../data/rna_pattern_4.txt", sep = "\t")
#df_5.to_csv("../data/rna_pattern_5.txt", sep = "\t")