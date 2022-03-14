import matplotlib.pyplot as plt
import seaborn as sns
import os

def view_class(data, col_name, idx2class, target_dir = None, file_name = None):
    data[col_name].replace(idx2class, inplace=True)
    #plt.figure(figsize=(6,3)) 
    plt.figure()
    #sns.set(font_scale = 1.2)
    sns.countplot(x = col_name, data = data).set(title="class distribution")
    if target_dir is not None and file_name is not None:
        plt.savefig(os.path.join(target_dir, file_name))
    else:
        return plt

