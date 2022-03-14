import pandas as pd
import pdb as pb

pancancer = pd.read_csv("data/pancancer_12_mRNA.csv", index_col=0)
selected = ["CESC", "CHOL", "COAD", "ESCA", "READ"]
subset = pancancer.loc[pancancer["cancer"].isin (selected)]
subset.to_csv("data/pancancer_5_mRNA.csv")










