import pandas as pd
import numpy as np

data = pd.read_csv('tcga_ref_diagnostic_age.csv')
data = data[data.Labels != "'--"]
ages = data['Labels'].values.astype(int)

ages = np.where((ages >= 10) & (ages < 20), 0, ages)
ages = np.where((ages >= 20) & (ages < 30), 1, ages)
ages = np.where((ages >= 30) & (ages < 40), 2, ages)
ages = np.where((ages >= 40) & (ages < 50), 3, ages)
ages = np.where((ages >= 50) & (ages < 60), 4, ages)
ages = np.where((ages >= 60) & (ages < 70), 5, ages)
ages = np.where((ages >= 70) & (ages < 80), 6, ages)
ages = np.where((ages >= 80) & (ages < 90), 7, ages)
ages = np.where((ages >= 90) & (ages <= 100), 8, ages)
import pdb; pdb.set_trace()
data['Labels'] = ages
data.to_csv('tcga_ref_diagnostoc_age_ordinal.csv', index=False)
