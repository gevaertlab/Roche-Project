import os
import json
import argparse
import datetime
import pickle

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn import preprocessing

from wsi_model import *
from read_data import *
from resnet import resnet50

parser = argparse.ArgumentParser(description='SSL training')
parser.add_argument('--config', type=str, help='JSON config file')
parser.add_argument('--checkpoint', type=str, default=None,
        help='File with the checkpoint to start with')
parser.add_argument('--seed', type=int, default=99,
        help='Seed for random generation')
parser.add_argument('--log', type=int, default=0,
        help='Use tensorboard for experiment logging')
parser.add_argument('--parallel', type=int, default=0,
        help='Use DataParallel training')
parser.add_argument('--fp16', type=int, default=0,
        help='Use mixed-precision training')

args = parser.parse_args()

np.random.seed(args.seed)
torch.manual_seed(args.seed)

with open(args.config) as f:
    config = json.load(f)

print(10*'-')
print('Config for this experiment \n')
print(config)
print(10*'-')

if 'flag' in config:
    args.flag = config['flag']
else:
    args.flag = 'train_{date:%Y-%m-%d %H:%M:%S}'.format(date=datetime.datetime.now())

if not os.path.exists(config['save_dir']):
    if not os.path.exists(config['save_dir'].split('/')[0]):
        os.mkdir(config['save_dir'].split('/')[0])
    os.mkdir(config['save_dir'])

path_csv = config['path_csv']
patch_data_path = config['patch_data_path']
img_size = config['img_size']
max_patch_per_wsi = config['max_patch_per_wsi']
quick = config.get('quick', None)
bag_size = config.get('bag_size', 40)
batch_size = config.get('batch_size', 64)
type_ = config.get('type', 'classification')

if 'quick' in config:
    quick = config['quick']
else:
    quick = None

transforms_val = torch.nn.Sequential(
    transforms.ConvertImageDtype(torch.float),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))

print('Loading dataset...')

df = pd.read_csv(path_csv)

_, test_df = split_train_test_ids(df, test_ids_path=config['test_ids'])

# getting encoding for labels
if type_ == 'classification':
    le = preprocessing.LabelEncoder()
    le.fit(config['classes'])
    print(list(le.classes_))
else:
    le = None

test_dataset = PatchBagDataset(patch_data_path, test_df, img_size,
                         max_patches_total=max_patch_per_wsi,
                         bag_size=bag_size,
                         transforms=transforms_val, quick=quick,
                         label_encoder=le)

if torch.cuda.is_available():
    #num_workers = torch.cuda.device_count() * 4
    num_workers = 10

test_dataloader = DataLoader(test_dataset,  
num_workers=num_workers, pin_memory=True, shuffle=False, batch_size=1)

print('Finished loading dataset and creating dataloader')

print('Initializing models')

resnet50 = resnet50(pretrained=True)

layers_to_train = [resnet50.fc, resnet50.layer4, resnet50.layer3]
for param in resnet50.parameters():
    param.requires_grad = False
for layer in layers_to_train:
    for n, param in layer.named_parameters():
        param.requires_grad = True

resnet50 = resnet50.to('cuda:0')
if type_ == 'classification':
    model = AggregationModel(resnet50, num_outputs=len(config['classes']))
else:
    model = AggregationModel(resnet50, num_outputs=1)

if args.checkpoint is not None:
    print('Restoring from checkpoint')
    print(args.checkpoint)
    model.load_state_dict(torch.load(args.checkpoint))
    print('Loaded model from checkpoint')
else:
    print('You need to load a checkpoint.')
    exit(0)

if args.parallel and torch.cuda.device_count() > 2:
    model = nn.DataParallel(model)

if torch.cuda.is_available():
    model.to('cuda:0')
# add optimizer
lr = config.get('lr', 3e-3)

optimizer = AdamW(model.parameters(), weight_decay = config['weights_decay'], lr=lr)

# add loss function
if type_ == 'classification':
    criterion = nn.CrossEntropyLoss()
else:
    criterion = nn.MSELoss()

# test on test set

test_results = evaluate(model, test_dataloader, len(test_dataset),
                            transforms_val, criterion=criterion,
                            problem=type_, device='cuda:0')

with open(config['save_dir']+'test_results.pkl', 'wb') as file:
    pickle.dump(test_results, file)
