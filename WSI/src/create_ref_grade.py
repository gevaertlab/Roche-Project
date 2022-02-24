import glob
import os
import pandas as pd
import pickle

diseases = ['PRAD']
grades = {
    'Pattern 3': 'Pattern 3',
    'Pattern 4': 'Pattern 4',
    'Pattern 5': 'Pattern 5'
}

def extract_code(label):
    return int(label.split(' ')[-1])

for disease in diseases:
    wsi_file_names = glob.glob('../TCGA-'+disease+'/*.svs')
    clinical = pd.read_csv('../TCGA-'+disease+'/clinical.tsv', sep='\t')
    wsi_names = []
    labels = []
    tcga_project = []
    error_label = []
    gene_info = pd.read_csv('../gene_expression/subset_pancer_mRNA.txt', sep=' ',
                        low_memory=False)
    genes = gene_info.index.values[:-1]
    genes_mr = [gene.replace('rna_', '') for gene in genes]
    prad_data = pd.read_csv('../gene_expression/PRAD_mRNA.txt', sep= ' ', low_memory=False)
    prad_data = prad_data[prad_data.index.isin(genes_mr)]
    genes = prad_data.index.values[:-1]
    genes_dict = {}
    for gene in genes:
        genes_dict['rna_'+gene] = []
    patient_ids_genes = prad_data.columns.values
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
                    primary = extract_code(row['primary_gleason_grade'].values[0])
                    secondary = extract_code(row['secondary_gleason_grade'].values[0])
                    label = 'Pattern {}'.format(primary+secondary)
                except:
                    #print('Error with {};{}'.format(name,label))
                    error_label.append(name)
                    continue
                wsi_names.append(name)
                labels.append(label)
                tcga_project.append('TCGA-'+disease)
                patient_ids.append(patient_id)
                column = prad_data[patient_id].values[:-1]
                for value, key in zip(column, genes):
                    genes_dict['rna_'+key].append(value)
    print('Number of wrong labels: {}'.format(len(error_label)))
    data = pd.DataFrame()
    data['wsi_file_name'] = wsi_names
    data['Labels'] = labels
    data['tcga_project'] = tcga_project

    genes_data = pd.DataFrame.from_dict(genes_dict)
    whole_data = pd.concat([data, genes_data], axis=1)
    whole_data.to_csv('ref_grades/tcga_ref_diagnostic_gleasonscore_'+disease+'_fusion.csv', index=False)

    #data.to_csv('ref_grades/tcga_ref_diagnostic_grade_'+disease+'.csv', index=False)
    with open('ref_grades/error_label_sec_grade_fusion_'+disease+'.pkl', 'wb') as f:
        pickle.dump(error_label, f)
