import os
import json
import argparse
import datetime
import pickle

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from torchvision import transforms
from sklearn.model_selection import train_test_split
from spacecutter.losses import CumulativeLinkLoss
from spacecutter.models import OrdinalLogisticModel
from warmup_scheduler import GradualWarmupScheduler

from wsi_model import *
from read_data import *
from resnet import resnet50
#from parallel import DataParallelModel, DataParallelCriterion

parser = argparse.ArgumentParser(description='Ordinal training')
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

transforms_ = torch.nn.Sequential(
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(64.0 / 255, 0.75, 0.25, 0.04),
    transforms.ConvertImageDtype(torch.float),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))

transforms_val = torch.nn.Sequential(
    transforms.ConvertImageDtype(torch.float),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))

print('Loading dataset...')

df = pd.read_csv(path_csv)

train_df, test_df = split_train_test_ids(df, test_ids_path=config['test_ids'])

train_df, val_df = train_test_split(train_df, test_size=0.2, stratify=train_df['Labels'])
le = None

train_dataset = PatchBagDataset(patch_data_path, train_df, img_size,
                         max_patches_total=max_patch_per_wsi,
                         bag_size=bag_size,
                         transforms=transforms_, quick=quick,
                         label_encoder=le, ordinal=True)
val_dataset = PatchBagDataset(patch_data_path, val_df, img_size,
                         max_patches_total=max_patch_per_wsi,
                         bag_size=bag_size,
                         transforms=transforms_val, quick=quick,
                         label_encoder=le, ordinal=True)

test_dataset = PatchBagDataset(patch_data_path, test_df, img_size,
                         max_patches_total=max_patch_per_wsi,
                         bag_size=bag_size,
                         transforms=transforms_val, quick=quick,
                         label_encoder=le, ordinal=True)

if torch.cuda.is_available():
    num_workers = config.get('num_workers', torch.cuda.device_count() * 4)

train_dataloader = DataLoader(train_dataset, 
            num_workers=num_workers, pin_memory=True, 
            shuffle=True, batch_size=batch_size)
val_dataloader = DataLoader(val_dataset, num_workers=num_workers,
pin_memory=True, shuffle=False, batch_size=batch_size)
test_dataloader = DataLoader(test_dataset,  
num_workers=num_workers, pin_memory=True, shuffle=False, batch_size=batch_size)

dataloaders = {
        'train': train_dataloader,
        'val': val_dataloader}

dataset_sizes = {
        'train': len(train_dataset),
        'val': len(val_dataset)
        }

transforms = {
        'train': transforms_,
        'val': transforms_val
        }
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

aggregator = AggregationModel(resnet50, num_outputs=1)

model = OrdinalLogisticModel(aggregator, len(config['classes']))

if args.checkpoint is not None:
    print('Restoring from checkpoint')
    print(args.checkpoint)
    model.load_state_dict(torch.load(args.checkpoint))
    print('Loaded model from checkpoint')

if args.parallel and torch.cuda.device_count() > 2:
    model = nn.DataParallel(model)
if torch.cuda.is_available():
    model.to('cuda:0')
# add optimizer
lr = config.get('lr', 3e-3)

optimizer = AdamW(model.parameters(), weight_decay = config['weights_decay'], lr=lr)

# add loss function
criterion = CumulativeLinkLoss()

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 500)
scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=1000, after_scheduler=scheduler)

# train model

if args.log:
    summary_writer = SummaryWriter(
            os.path.join(config['summary_path'],
                datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S") + "_{0}".format(args.flag)))

    summary_writer.add_text('config', str(config))
else:
    summary_writer = None

model, results = train(model, criterion, optimizer, dataloaders, transforms, 
              save_dir=config['save_dir'],
              device=config['device'], log_interval=config['log_interval'],
              summary_writer=summary_writer,
              num_epochs=config['num_epochs'],
              problem=type_,
              scheduler=scheduler_warmup)

# test on test set

test_results = evaluate(model, test_dataloader, len(test_dataset),
                            transforms_val, criterion=criterion, device=config['device'],
                            problem=type_)

with open(config['save_dir']+'test_results.pkl', 'wb') as file:
    pickle.dump(test_results, file)
