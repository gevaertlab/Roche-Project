from unicodedata import numeric
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from statannotations.Annotator import Annotator
from util import format_graph, construct_pairs
import datetime
import argparse
import pdb
import json
import os
import sys


''' Initiate configuration '''
parser = argparse.ArgumentParser(description='SSL training')
parser.add_argument('--config', type=str, help='JSON config file')
parser.add_argument('--seed', type=int, default=99,
        help='Seed for random generation')
parser.add_argument('--flag', type=str, default=None,
        help='Flag to use for saving the checkpoints')

args = parser.parse_args()
with open(args.config) as f:
    config = json.load(f)

print(10*'-')
print('Config for this experiment \n')
print(config)
print(10*'-')

print(10*'-')
print('Args for this experiment \n')
print(args)
print(10*'-')

clinical_data = config['clinical_data']
cell_data = config['cell_data']
output_dir = config['output_dir']
x_col = config["x_axis"]
x_label = config['x_label']
y_label = config['y_label']
plot_type = config["plot_type"]
test_method = config['test_method']
adjusted_method = config['adjusted_method']
generate_group_graph = config['generate_group_graph']
generate_seperate_graph = config["generate_seperate_graph"]

summary_file = os.path.join(output_dir, 'Summary_{date:%Y-%m-%d}'.format(date = datetime.datetime.now()) + '.txt')

print("Reading in data")
# Clinical data with the row index as the patient names, and a column ("x_col") indicating the groups
clinical_df = pd.read_csv(clinical_data, index_col = 0)
clinical_df["patient_id"] = clinical_df.index.values.tolist()

# Cibersort retulsts with the row index as patient names, and columns represent different cell types
result_df = pd.read_csv(cell_data, index_col = 0)
result_df['patient_id'] = result_df.index.values.tolist()

# Join clinical data and Cibersort results
merged_df = result_df.merge(clinical_df[["patient_id", "primary_gleason_grade"]], on = "patient_id", how = "inner")
merged_df = merged_df.drop(["Correlation","RMSE", "P-value", "patient_id"], axis=1)
data = merged_df.melt(x_col, var_name='Cell_type', value_name='Proportion')
data['Percentage'] = data["Proportion"] * 100

if generate_seperate_graph == "True":
    # Iteration through different cell types and generate an annotated bar graph for each 
    cell_types = data["Cell_type"].unique()
    for cell in cell_types:
        print(cell)
        cur_data = data[data["Cell_type"]==cell]
        order = ["3", "4", "5"]
        cur_data = cur_data.astype({x_col: str})
        plotting_parameters = {
            'data': cur_data,
            'x' : x_col,
            'y' : "Percentage",
            'order': order
        }
        pairs = [('3', '4'),('4', '5'),('3', '5')]

        # Generate the plot
        fig = plt.figure(figsize=(7,6))
        sns.set(font_scale = 1.5)
        ax = sns.violinplot(**plotting_parameters)
        format_graph(ax = ax, title = cell, x_label = x_label, y_label = y_label)
        #change_width(ax, 0.35)

        # Add statistical annotation
        annotator = Annotator(ax = ax, pairs = pairs, plot = "violinplot",  **plotting_parameters)
        annotator.configure(test=test_method, comparisons_correction=adjusted_method, verbose=False, text_format='star')
        _, corrected_results = annotator.apply_and_annotate()
        plt.savefig(os.path.join(output_dir, cell  + '_{date:%Y-%m-%d}'.format(date = datetime.datetime.now()) + '.png'))

# if(generate_group_graph == "True"):
#     # Generate a grouped picture 
#     fig = plt.figure()
#     order = list(data["Cell_type"].unique())
#     hue_order = ['3', '4', '5']
#     pdb.set_trace()
#     pairs = construct_pairs(order, hue_order)
#     print(pairs)
#     hue_plotting_params = {
#         'data': data,
#         'x' : x_col,
#         'y' : "Percentage",
#         'order': order,
#         'hue': "Cell_type",
#         'hue_order':hue_order
#     }
#     ax = sns.violinplot(**hue_plotting_params)
#     annotator = Annotator(ax = ax, pairs = pairs, plot = "violinplot",  **hue_plotting_params)
#     annotator.configure(test=test_method, comparisons_correction=adjusted_method, verbose=False, text_format='star')
#     _, corrected_results = annotator.apply_and_annotate()
#     plt.savefig(os.path.join(output_dir,  + 'All_types_{date:%Y-%m-%d}'.format(date = datetime.datetime.now()) + '.png'))


# # Kruscal test
# summary_writter = "Results/sig.txt"
# f = open(summary_writter, 'w')
# types = data["Cell type"].unique()
# for cell in types:
#     cur_data = data[data["Cell type"]==cell]
#     cur_array = [cur_data.loc[ids, 'Percentage'].values for ids in cur_data.groupby('primary_gleason_grade').groups.values()]
#     H, p = ss.kruskal(*cur_array)
#     if p < 0.05:
#         print(cell)
#         print(p)

#         # draw the graph
#         #fig = plt.figure(figsize=(8,6))
#         fig = plot.figure()
#         sns.set(font_scale = 1.5)
#         ax=sns.barplot(x = "primary_gleason_grade", y='Percentage', data=cur_data)
#         ax.set(title=cell)
#         plt.savefig("Results/" + cell + ".png")

#         # compare individual groups and save the results
#         sig = sp.posthoc_conover(cur_data, val_col='Percentage', group_col='primary_gleason_grade', p_adjust = 'holm')
#         print(sig)
#         with open(summary_writter, 'a+') as f:
#             dfAsString = sig.to_string(header=True, index=True)
#             f.write(cell + "\n")
#             f.write(str(p) + "\n")
#             f.write(dfAsString + "\n")

    

