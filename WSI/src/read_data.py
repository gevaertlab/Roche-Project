import os
import pickle
import random

import numpy as np
from torch.utils.data import Dataset
import pandas as pd
import torch
from tqdm import tqdm
import lmdb
import lz4framed

from typing import Any

class PatchBagDataset(Dataset):
    def __init__(self, patch_data_path, csv_path, img_size, transforms=None, bag_size=40,
            max_patches_total=300, quick=False, label_encoder=None, ordinal=False, 
            type='classification'):
        self.patch_data_path = patch_data_path
        self.csv_path = csv_path
        self.img_size = img_size
        self.transforms = transforms
        self.bag_size = bag_size
        self.max_patches_total = max_patches_total
        self.quick = quick
        self.le = label_encoder
        self.ordinal = ordinal
        self.type = type
        self.index = []
        self.data = {}
        self._preprocess()

    def _preprocess(self):
        if type(self.csv_path) == str:
            csv_file = pd.read_csv(self.csv_path)
        else:
            csv_file = self.csv_path
        
        if self.quick:
            csv_file = csv_file.sample(150)
        
        for i, row in tqdm(csv_file.iterrows()):
            row = row.to_dict()
            WSI = row['wsi_file_name']
            label = np.asarray(row['Labels'])
            if self.le is not None:
                label = self.le.transform(label.reshape(-1,1))
                if self.type == 'regression':
                    label = label.astype(np.float32)
            else:
                if self.ordinal:
                    label = label.astype(np.int64)
                else:
                    label = label.astype(np.float32)

            project = row['tcga_project'] 
            if not os.path.exists(os.path.join('../'+project+self.patch_data_path, WSI)):
                print('Not exist {}'.format(os.path.join('../'+project+self.patch_data_path, WSI)))
                continue
            
            #try:
            path = os.path.join('../'+project+self.patch_data_path, WSI, WSI)
            try:
                lmdb_connection = lmdb.open(path,
                                            subdir=False, readonly=True, 
                                            lock=False, readahead=False, meminit=False)
            except:
                path = path + '.db'
                try:
                    lmdb_connection = lmdb.open(path,
                                                subdir=False, readonly=True, 
                                                lock=False, readahead=False, meminit=False)
                except:
                    continue
            try:
                with lmdb_connection.begin(write=False) as lmdb_txn:
                    n_patches = lmdb_txn.stat()['entries'] - 1
                    keys = pickle.loads(lz4framed.decompress(lmdb_txn.get(b'__keys__')))
            except Exception as e:
                print(e)
                continue
            #except:
            #    print('Error with lmdb file {}'.format(os.path.join('../'+project+self.patch_data_path, WSI)))
            #    continue
            n_selected = min(n_patches, self.max_patches_total)
            n_patches= list(range(n_patches))
            images = random.sample(n_patches, n_selected)
            self.data[WSI] = {w.lower(): row[w] for w in row.keys()}
            self.data[WSI].update({'WSI': WSI, 'images': images, 'n_images': len(images), 
                                   'lmdb_path': path, 'keys': keys})
            for k in range(len(images) // self.bag_size):
                self.index.append((WSI, self.bag_size * k, label))

    def shuffle(self):
        for k in self.data.keys():
            wsi_row = self.data[k]
            np.random.shuffle(wsi_row['images'])

    def decompress_and_deserialize(self, lmdb_value: Any):
        try:
            img_name, img_arr, img_shape = pickle.loads(lz4framed.decompress(lmdb_value))
        except:
            return None
        image = np.frombuffer(img_arr, dtype=np.uint8).reshape(img_shape)
        image = np.copy(image)
        return torch.from_numpy(image).permute(2,0,1)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        (WSI, i, label) = self.index[idx]
        imgs = []
        row = self.data[WSI]
        lmdb_connection = lmdb.open(row['lmdb_path'],
                                        subdir=False, readonly=True, 
                                        lock=False, readahead=False, meminit=False)
        with lmdb_connection.begin(write=False) as txn:
            for patch in row['images'][i:i + self.bag_size]:
                lmdb_value = txn.get(row['keys'][patch])
                img = self.decompress_and_deserialize(lmdb_value)
                imgs.append(img)

        img = torch.stack(imgs, dim=0)
        return img, label


class PatchBagRNADataset(Dataset):
    def __init__(self, patch_data_path: str, csv_path: str, img_size:int , 
                    transforms=None, max_patch_per_wsi=400, bag_size=20,
                    quick=None, label_encoder=None, type='classification',
                    ordinal=False):
        self.patch_data_path = patch_data_path
        self.csv_path = csv_path
        self.img_size = img_size
        self.bag_size = bag_size
        self.transforms = transforms
        self.max_patches_total = max_patch_per_wsi
        self.quick = quick
        self.le = label_encoder
        self.ordinal = ordinal
        self.type = type
        self.data = {}
        self.index = []
        self._preprocess()

    def _preprocess(self):
        if type(self.csv_path) == str:
            csv_file = pd.read_csv(self.csv_path)
        else:
            csv_file = self.csv_path
        
        if self.quick:
            csv_file = csv_file.sample(150)
        
        for i, row in tqdm(csv_file.iterrows()):
            WSI = row['wsi_file_name']
            label = np.asarray(row['Labels'])
            if label == "'--": continue
            rna_data = row[[x for x in row.keys() if 'rna_' in x]].values.astype(np.float32)
            rna_data = torch.tensor(rna_data, dtype=torch.float32)
            if self.le is not None:
                label = self.le.transform(label.reshape(-1,1))
                if self.type == 'regression':
                    label = label.astype(np.float32)
            else:
                if self.ordinal:
                    label = label.astype(np.int64)
                else:
                    label = label.astype(np.float32)

            project = row['tcga_project'] 
            if not os.path.exists(os.path.join('../'+project+self.patch_data_path, WSI)):
                print('Not exist {}'.format(os.path.join('../'+project+self.patch_data_path, WSI)))
                continue
            
            #try:
            path = os.path.join('../'+project+self.patch_data_path, WSI, WSI)
            try:
                lmdb_connection = lmdb.open(path,
                                            subdir=False, readonly=True, 
                                            lock=False, readahead=False, meminit=False)
            except:
                path = path + '.db'
                try:
                    lmdb_connection = lmdb.open(path,
                                                subdir=False, readonly=True, 
                                                lock=False, readahead=False, meminit=False)
                except:
                    continue
            try:
                with lmdb_connection.begin(write=False) as lmdb_txn:
                    n_patches = lmdb_txn.stat()['entries'] - 1
                    keys = pickle.loads(lz4framed.decompress(lmdb_txn.get(b'__keys__')))
            except Exception as e:
                print(e)
                continue

            #except:
            #    print('Error with loc file {}'.format(os.path.join('../'+project+self.patch_data_path, WSI)))
            #    continue
            n_selected = min(n_patches, self.max_patches_total)
            n_patches= list(range(n_patches))
            images = random.sample(n_patches, n_selected)
            new_row = dict()
            new_row['WSI'] = WSI
            new_row['rna_data'] = rna_data
            new_row['label'] = label
            self.data[WSI] = {w.lower(): new_row[w] for w in new_row.keys()}
            self.data[WSI].update({'WSI': WSI, 'images': images, 'n_images': len(images),
                                   'lmdb_path': path, 'keys': keys})
            for k in range(len(images) // self.bag_size):
                self.index.append((WSI, self.bag_size * k))

    def decompress_and_deserialize(self, lmdb_value: Any):
        try:
            img_name, img_arr, img_shape = pickle.loads(lz4framed.decompress(lmdb_value))
        except:
            return None
        image = np.frombuffer(img_arr, dtype=np.uint8).reshape(img_shape)
        image = np.copy(image)
        return torch.from_numpy(image).permute(2,0,1)

    def __len__(self):
        return len(self.index)
    
    def __getitem__(self, idx):
        (WSI, i) = self.index[idx]
        imgs = []
        row = self.data[WSI].copy()
        lmdb_connection = lmdb.open(row['lmdb_path'],
                                        subdir=False, readonly=True, 
                                        lock=False, readahead=False, meminit=False)
        with lmdb_connection.begin(write=False) as txn:
            for patch in row['images'][i:i + self.bag_size]:
                lmdb_value = txn.get(row['keys'][patch])
                img = self.decompress_and_deserialize(lmdb_value)
                imgs.append(img)
        img = torch.stack(imgs,dim=0)
        return img, row['rna_data'], row['label']

class PandaPatchBagDataset(Dataset):
    def __init__(self, patch_data_path, csv_path, img_size, transforms=None, bag_size=40,
            max_patches_total=300, quick=False, label_encoder=None, ordinal=False, 
            type='classification'):
        self.patch_data_path = patch_data_path
        self.csv_path = csv_path
        self.img_size = img_size
        self.transforms = transforms
        self.bag_size = bag_size
        self.max_patches_total = max_patches_total
        self.quick = quick
        self.le = label_encoder
        self.ordinal = ordinal
        self.type = type
        self.index = []
        self.data = {}
        self._preprocess()

    def _preprocess(self):
        if type(self.csv_path) == str:
            csv_file = pd.read_csv(self.csv_path)
        else:
            csv_file = self.csv_path
        
        if self.quick:
            csv_file = csv_file.sample(150)
        
        for i, row in tqdm(csv_file.iterrows()):
            row = row.to_dict()
            WSI = row['image_id']
            if not os.path.exists(os.path.join('../pandas-dataset/panda_patches', WSI)):
                continue
            label = np.asarray(row['gleason_score'].split('+')[0])
            if self.le is not None:
                if label not in list(self.le.classes_): continue
                label = self.le.transform(label.reshape(-1,1))
                if self.type == 'regression':
                    label = label.astype(np.float32)
            else:
                if self.ordinal:
                    label = label.astype(np.int64)
                else:
                    label = label.astype(np.float32)
            
            #try:
            path = os.path.join('../pandas-dataset/panda_patches', WSI, WSI)
            try:
                lmdb_connection = lmdb.open(path,
                                            subdir=False, readonly=True, 
                                            lock=False, readahead=False, meminit=False)
            except:
                path = path + '.db'
                try:
                    lmdb_connection = lmdb.open(path,
                                                subdir=False, readonly=True, 
                                                lock=False, readahead=False, meminit=False)
                except:
                    continue
            try:
                with lmdb_connection.begin(write=False) as lmdb_txn:
                    n_patches = lmdb_txn.stat()['entries'] - 1
                    keys = pickle.loads(lz4framed.decompress(lmdb_txn.get(b'__keys__')))
            except Exception as e:
                continue
            #except:
            #    print('Error with lmdb file {}'.format(os.path.join('../'+project+self.patch_data_path, WSI)))
            #    continue
            n_selected = min(n_patches, self.max_patches_total)
            n_patches= list(range(n_patches))
            images = random.sample(n_patches, n_selected)
            self.data[WSI] = {w.lower(): row[w] for w in row.keys()}
            self.data[WSI].update({'WSI': WSI, 'images': images, 'n_images': len(images), 
                                   'lmdb_path': path, 'keys': keys})
            for k in range(len(images) // self.bag_size):
                self.index.append((WSI, self.bag_size * k, label))

    def shuffle(self):
        for k in self.data.keys():
            wsi_row = self.data[k]
            np.random.shuffle(wsi_row['images'])

    def decompress_and_deserialize(self, lmdb_value: Any):
        try:
            img_name, img_arr, img_shape = pickle.loads(lz4framed.decompress(lmdb_value))
        except:
            return None
        image = np.frombuffer(img_arr, dtype=np.uint8).reshape(img_shape)
        image = np.copy(image)
        return torch.from_numpy(image).permute(2,0,1)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        (WSI, i, label) = self.index[idx]
        imgs = []
        row = self.data[WSI]
        lmdb_connection = lmdb.open(row['lmdb_path'],
                                        subdir=False, readonly=True, 
                                        lock=False, readahead=False, meminit=False)
        with lmdb_connection.begin(write=False) as txn:
            for patch in row['images'][i:i + self.bag_size]:
                lmdb_value = txn.get(row['keys'][patch])
                img = self.decompress_and_deserialize(lmdb_value)
                imgs.append(img)

        img = torch.stack(imgs, dim=0)
        return img, torch.from_numpy(label).long()

def split_train_test_ids(df, test_ids_path):
    file_parse = np.loadtxt(test_ids_path, dtype=str)
    test_ids = [x.split('"')[1] for x in file_parse]
    wsi_file_names = df['wsi_file_name'].values

    test_wsi = []
    for test_id in test_ids:
        if test_id in wsi_file_names:
            test_wsi.append(test_id)
    
    test_df = df.loc[df['wsi_file_name'].isin(test_wsi)]
    train_df = df.loc[~df['wsi_file_name'].isin(test_wsi)]

    return train_df, test_df