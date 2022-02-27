import glob

import pandas as pd

diseases = ['ESCA', 'GBM', 'KIRP', 'LUAD', 'OV', 'PAAD', 'READ', 'UCS', 'UVM']
tissues = {
    'ESCA': 'Eshopagus',
    'GBM': 'Brain',
    'KIRP': 'Kidney',
    'LUAD': 'Lung',
    'OV': 'Ovary',
    'PAAD': 'Pancreas',
    'READ': 'Rectum',
    'UCS': 'Uterus',
    'UVM': 'Eyes'
}

wsi_names = []
labels = []
tcga_project = []
gene_info = pd.read_csv('../gene_expression/subset_pancer_mRNA.txt', sep=' ',
                        low_memory=False)
genes = gene_info.index.values[:-1]
genes_dict = {}
for gene in genes:
    genes_dict[gene] = []
patient_ids_genes = gene_info.columns.values
patient_ids = []
for disease in diseases:
    wsi_file_names = glob.glob('../TCGA-'+disease+'/*.svs')
    clinical = pd.read_csv('../TCGA-'+disease+'/clinical.tsv', sep='\t')
    patient_ids = []
    for x in wsi_file_names:
        name = x.split('/')[-1].replace('.svs', '')
        split_name = name.split('-')
        patient_id = split_name[0] + '-' + split_name[1] + '-' + split_name[2]
        if '-DX' in name:
            if patient_id in patient_ids_genes:
                id = name.split('-')[0] + '-' + name.split('-')[1] + '-' + name.split('-')[2]
                row = clinical.loc[clinical['case_submitter_id'] == id]
                try:
                    label = row['age_at_index'].values[0]
                except:
                    print('Error with {}'.format(name))
                    continue
                wsi_names.append(name)
                labels.append(label)
                tcga_project.append('TCGA-'+disease)
                patient_ids.append(patient_id)
                column = gene_info[patient_id].values[:-1]
                for value, key in zip(column, genes):
                    genes_dict[key].append(value)

data = pd.DataFrame()
data['wsi_file_name'] = wsi_names
data['Labels'] = labels
data['tcga_project'] = tcga_project
genes_data = pd.DataFrame.from_dict(genes_dict)
whole_data = pd.concat([data, genes_data], axis=1)
whole_data.to_csv('tcga_ref_diagnostic_age_fusion.csv', index=False)
