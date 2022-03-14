import glob
import os
import pandas as pd
import pickle

diseases = ['ESCA', 'GBM', 'KIRP', 'LUAD', 'OV', 'PAAD', 'READ', 'UCS', 'UVM']
#diseases = ['PRAD']
stages = {
    'Stage I': 'Stage I',
    'Stage IA': 'Stage I',
    'Stage IA1': 'Stage I',
    'Stage IA2': 'Stage I',
    'Stage IA3': 'Stage I',
    'Stage IB': 'Stage I',
    'Stage IB1': 'Stage I',
    'Stage IB2': 'Stage I',
    'Stage IC': 'Stage I',
    'Stage II': 'Stage II',
    'Stage IIA': 'Stage II',
    'Stage IIA1': 'Stage II',
    'Stage IIA2': 'Stage II',
    'Stage IIB': 'Stage II',
    'Stage IIC': 'Stage II',
    'Stage IIC1': 'Stage II',
    'Stage III': 'Stage III',
    'Stage IIIA': 'Stage III',
    'Stage IIIB': 'Stage III',
    'Stage IIIC': 'Stage III',
    'Stage IIIC1': 'Stage III',
    'Stage IIIC2': 'Stage III',
    'Stage IV': 'Stage IV',
    'Stage IVA': 'Stage IV',
    'Stage IVB': 'Stage IV',
    'Stage IVC': 'Stage IV',
}


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
    genes_dict = {}
    for gene in genes:
        genes_dict[gene] = []
    patient_ids_genes = gene_info.columns.values
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
                    label = row['ajcc_pathologic_stage'].values[0]
                    label = stages[label]
                except:
                    #print('Error with {};{}'.format(name,label))
                    error_label.append(name)
                    continue
                wsi_names.append(name)
                labels.append(label)
                tcga_project.append('TCGA-'+disease)
                patient_ids.append(patient_id)
                column = gene_info[patient_id].values[:-1]
                for value, key in zip(column, genes):
                    genes_dict[key].append(value)
    
    print('Number of wrong labels: {}'.format(len(error_label)))
    data = pd.DataFrame()
    data['wsi_file_name'] = wsi_names
    data['Labels'] = labels
    data['tcga_project'] = tcga_project

    genes_data = pd.DataFrame.from_dict(genes_dict)
    whole_data = pd.concat([data, genes_data], axis=1)
    whole_data.to_csv('ref_stages/tcga_ref_diagnostic_stage_fusion'+disease+'.csv', index=False)
    #data.to_csv('ref_stages/tcga_ref_diagnostic_stage_'+disease+'.csv', index=False)
    with open('ref_stages/error_label_stage_fusion_'+disease+'.pkl', 'wb') as f:
        pickle.dump(error_label, f)
