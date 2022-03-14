import os
import json
import argparse
import datetime
import pickle

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, WeightedRandomSampler
from tensorboardX import SummaryWriter
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from warmup_scheduler import GradualWarmupScheduler
from sklearn.utils.class_weight import compute_class_weight

from wsi_model import *
from read_data import *
from resnet import resnet50

parser = argparse.ArgumentParser(description='SSL training')
parser.add_argument('--config', type=str, help='JSON config file')
parser.add_argument('--checkpoint', type=str, default=None,
        help='File with the checkpoint to start with')
parser.add_argument('--save_dir', type=str, default=None,
        help='Where to save the checkpoints')
parser.add_argument('--flag', type=str, default=None,
        help='Flag to use for saving the checkpoints')
parser.add_argument('--seed', type=int, default=99,
        help='Seed for random generation')
parser.add_argument('--log', type=int, default=0,
        help='Use tensorboard for experiment logging')
parser.add_argument('--parallel', type=int, default=0,
        help='Use DataParallel training')
parser.add_argument('--fp16', type=int, default=0,
        help='Use mixed-precision training')
parser.add_argument('--bag_size', type=int, default=50,
                    help='Bag size to use')
parser.add_argument('--max_patch_per_wsi', type=int, default=100,
                    help='Maximum number of paches per wsi')
parser.add_argument('--labels_reg', type=int, default=0,
        help='If you want to use labels for the regression problem')
parser.add_argument("--class_weights", help="if class weights want to be used",
                    action="store_true")
parser.add_argument("--focal_loss", help="if focal_loss wanted to be used",
                    action="store_true")
parser.add_argument("--attention", help="if using attention in the aggregation model",
                    action="store_true")
parser.add_argument("--over_sampling", help="if over_sampling is applied to the minority class",
                    action="store_true")

args = parser.parse_args()

np.random.seed(args.seed)
torch.manual_seed(args.seed)

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

if not args.flag:
    args.flag = 'train_{date:%Y-%m-%d %H:%M:%S}'.format(date=datetime.datetime.now())

if not os.path.exists(args.save_dir):
    if not os.path.exists(args.save_dir.split('/')[0]):
        os.mkdir(args.save_dir.split('/')[0])
    os.mkdir(args.save_dir)

path_csv = config['path_csv']
patch_data_path = config['patch_data_path']
img_size = config['img_size']
max_patch_per_wsi = args.max_patch_per_wsi
quick = config.get('quick', None)
bag_size = args.bag_size
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

if type_ == 'classification' or args.labels_reg == True:
    train_df, val_df = train_test_split(train_df, test_size=0.2, stratify=train_df['Labels'])
    if args.class_weights or args.over_sampling:
        class_weights = compute_class_weight('balanced', classes=np.unique(train_df['Labels']), 
                                             y=train_df['Labels'])
        weights= torch.tensor(class_weights,dtype=torch.float).to(config['device'])
        print(weights)
        if args.over_sampling:
            sampler = WeightedRandomSampler(
                weights=class_weights,
                num_samples=len(train_df),
                replacement=True)
    le = preprocessing.LabelEncoder()
    le.fit(config['classes'])
    print(list(le.classes_))
else:
    train_df, val_df = train_test_split(train_df, test_size=0.2)
    le = None

train_dataset = PatchBagDataset(patch_data_path, train_df, img_size,
                         max_patches_total=max_patch_per_wsi,
                         bag_size=bag_size,
                         transforms=transforms_, quick=quick,
                         label_encoder=le,
                         type=type_)
val_dataset = PatchBagDataset(patch_data_path, val_df, img_size,
                         max_patches_total=max_patch_per_wsi,
                         bag_size=bag_size,
                         transforms=transforms_val, quick=quick,
                         label_encoder=le,
                         type=type_)

test_dataset = PatchBagDataset(patch_data_path, test_df, img_size,
                         max_patches_total=max_patch_per_wsi,
                         bag_size=bag_size,
                         transforms=transforms_val, quick=quick,
                         label_encoder=le,
                         type=type_)
if torch.cuda.is_available():
    print('There is a GPU!')
    num_workers = config.get('num_workers', torch.cuda.device_count() * 4)

if args.over_sampling:
    train_dataloader = DataLoader(train_dataset, 
                num_workers=num_workers, pin_memory=True, 
                shuffle=False, batch_size=batch_size, sampler=sampler)
else:
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
if type_ == 'classification':
    if args.attention:
        model = AggregationModelAttention(resnet50, num_outputs=len(config['classes']))
        use_attention = True
    else:
        model = AggregationModel(resnet50, num_outputs=len(config['classes']))
        use_attention = False
elif type_ == 'regression':
    if args.attention:
        model = AggregationModelAttention(resnet50, num_outputs=1)
        use_attention = True
    else:
        model = AggregationModel(resnet50, num_outputs=1)
        use_attention = False

if args.checkpoint is not None:
    print('Restoring from checkpoint')
    print(args.checkpoint)
    model.load_state_dict(torch.load(args.checkpoint))
    print('Loaded model from checkpoint')

if args.parallel and torch.cuda.device_count() > 2:
    model = nn.DataParallel(model)
if torch.cuda.is_available():
    model = model.to(config['device'])
# add optimizer
lr = config.get('lr', 3e-3)

optimizer = AdamW(model.parameters(), weight_decay = config['weights_decay'], lr=lr)

# add loss function
if type_ == 'classification':
    if args.focal_loss:
        criterion = 'focal_loss'
    else:
        if args.class_weights:
            criterion = nn.CrossEntropyLoss(weight=weights)
        else:
            criterion = nn.CrossEntropyLoss()
elif type_ == 'regression':
    criterion = nn.MSELoss()

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

# train model
model, results = train(model, criterion, optimizer, dataloaders, transforms, 
              save_dir=args.save_dir,
              device=config['device'], log_interval=config['log_interval'],
              summary_writer=summary_writer,
              num_epochs=config['num_epochs'],
              problem=type_,
              scheduler=scheduler_warmup,
              use_attention=use_attention)

with open(args.save_dir+'train_val_results.pkl', 'wb') as file:
    pickle.dump(results, file)

# test on test set

test_results = evaluate(model, test_dataloader, len(test_dataset),
                            transforms_val, criterion=criterion, device=config['device'],
                            problem=type_, use_attention=use_attention)

with open(args.save_dir+'test_results.pkl', 'wb') as file:
    pickle.dump(test_results, file)
