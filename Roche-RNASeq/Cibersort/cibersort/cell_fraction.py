from enum import unique
import pandas as pd
import pdb
import scipy.stats as ss
import scikit_posthocs as sp
import statsmodels.api as sa
import seaborn as sns
from matplotlib import pyplot as plt
#from statannotations.Annotator import Annotator

print("Reading in data")
clinical_df = pd.read_csv("../PASNet-prostate/data/PRAD_rna_clinical.csv", index_col = 0)
clinical_df["patient_id"] = clinical_df.index.values.tolist()
result_df = pd.read_csv("Results/CIBERSORTx_Job4_Results.csv", index_col = 0)
result_df['patient_id'] = result_df.index.values.tolist()

merged_df = result_df.merge(clinical_df[["patient_id", "primary_gleason_grade"]], on = "patient_id", how = "inner")
merged_df = merged_df.drop(["Correlation","RMSE", "P-value", "patient_id"], axis=1)

#merged_df.to_csv("Results/cell_frac.csv")

data = merged_df.melt('primary_gleason_grade', var_name='Cell type', value_name='Proportion')
data["Percentage"] = data["Proportion"]*100
# fig = plt.figure(figsize=(30,20))
# sns.set(font_scale = 1.2)
# sns.set_style("white")
# ax=sns.barplot(y='Cell type', x='Percentage', hue='primary_gleason_grade', data=data)
# plt.savefig("Results/frac.png")

# Kruscal test
summary_writter = "Results/sig.txt"
f = open(summary_writter, 'w')
types = data["Cell type"].unique()
for cell in types:
    cur_data = data[data["Cell type"]==cell]
    cur_array = [cur_data.loc[ids, 'Percentage'].values for ids in cur_data.groupby('primary_gleason_grade').groups.values()]
    H, p = ss.kruskal(*cur_array)
    if p < 0.05:
        print(cell)
        print(p)

        # draw the graph
        fig = plt.figure(figsize=(8,6))
        sns.set(font_scale = 1.5)
        ax=sns.barplot(x = "primary_gleason_grade", y='Percentage', data=cur_data)
        ax.set(title=cell)
        plt.savefig("Results/" + cell + ".png")

        # compare individual groups and save the results
        sig = sp.posthoc_conover(cur_data, val_col='Percentage', group_col='primary_gleason_grade', p_adjust = 'holm')
        print(sig)
        with open(summary_writter, 'a+') as f:
            dfAsString = sig.to_string(header=True, index=True)
            f.write(cell + "\n")
            f.write(str(p) + "\n")
            f.write(dfAsString + "\n")

    

