import torch
import torch.nn as nn
import os
import numpy as np
from tqdm import tqdm

torch.backends.cudnn.benchmark = True

def train_fusion(model, criterion, optimizer, dataloaders, transforms,
          save_dir='checkpoints/models/', device='cpu',
          log_interval=100, summary_writer=None, num_epochs=100, 
          problem='classification', scheduler=None, verbose=True):
    """ 
    Train classification/regression fusion model.
        Parameters:
            model (torch.nn.Module): Pytorch model already declared.
            criterion (torch.nn): Loss function
            optimizer (torch.optim): Optimizer
            dataloaders (dict): dict containing training and validation DataLoaders
            transforms (dict): dict containing training and validation transforms
            save_dir (str): directory to save checkpoints and models.
            device (str): device to move models and data to.
            log_interval (int): 
            summary_writer (TensorboardX): to register values into tensorboard
            num_epochs (int): number of epochs of the training
            problem (str): if it is a classification or regresion problem
            verbose (bool): wether or not to display metrics during training

        Returns:
            test_results (dict): dictionary containing the labels, predictions,
                                 probabilities and accuracy of the model on the dataset.
    """
    best_acc = 0.0
    best_epoch = 0
    best_loss = np.inf

    acc_array = {'train': [], 'val': []}
    loss_array = {'train': [], 'val': []}
    
    global_summary_step = {'train': 0, 'val': 0}

    # Creates once at the beginning of training
    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        sizes = {'train': 0, 'val': 0}
        inputs_seen = {'train': 0, 'val': 0}
    

        for phase in ['train', 'val']:
            if phase == 'train':
                    model.train()
            else:
                    model.eval()

            running_loss = 0.0
            if problem == 'classification' or problem == 'ordinal':
                running_corrects = 0.0
            summary_step = global_summary_step[phase]
            # for logging tensorboard
            last_running_loss = 0.0
            if problem == 'classification' or problem=='ordinal':
                last_running_corrects = 0.0
            for batch in tqdm(dataloaders[phase]):
                wsi = batch[0]
                rna = batch[1]
                labels = batch[2]
                size = wsi.size(0)

                if problem == 'classification':
                    labels = labels.flatten()
                labels = labels.to(device)
                #wsi = wsi.to(device)
                #rna = rna.to(device)
                wsi = transforms[phase](wsi)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase=='train'):
                    # Casts operations to mixed precision
                    with torch.cuda.amp.autocast():
                        outputs, embd1, embd2 = model(rna, wsi)
                        if problem == 'classification':
                            _, preds = torch.max(outputs,1)
                            loss = criterion(outputs, labels)
                        elif problem == 'regression':
                            loss = criterion(outputs, labels.view(labels.size(0), 1))
                        elif problem == 'ordinal':
                            _, preds = torch.max(outputs,1)
                            loss = criterion(outputs, labels.view(labels.size(0), 1))

                    if phase == 'train':
                        # Scales the loss, and calls backward()
                        # to create scaled gradients
                        scaler.scale(loss).backward()
                        
                        # Unscales gradients and calls
                        # or skips optimizer.step()
                        scaler.step(optimizer)
                        
                        # Updates the scale for next iteration
                        scaler.update()
                        if scheduler is not None:
                            scheduler.step()

                summary_step += 1
                running_loss += loss.item() * wsi.size(0)
                if problem == 'classification' or problem == 'ordinal':
                    running_corrects += torch.sum(preds == labels)
                sizes[phase] += size
                inputs_seen[phase] += size

                # Emptying memory
                outputs = outputs.detach()
                loss = loss.detach()
                torch.cuda.empty_cache()

                if (summary_step % log_interval == 0):
                    loss_to_log = (running_loss - last_running_loss) / inputs_seen[phase]
                    if problem == 'classification' or problem == 'ordinal':
                        acc_to_log = (running_corrects - last_running_corrects) / inputs_seen[phase]

                    if summary_writer is not None:
                        summary_writer.add_scalar("{}/loss".format(phase), loss_to_log, summary_step)
                        if problem == 'classification' or 'problem' == 'ordinal':
                            summary_writer.add_scalar("{}/acc".format(phase), acc_to_log, summary_step)

                    last_running_loss = running_loss
                    if problem == 'classification' or problem == 'ordinal':
                        last_running_corrects = running_corrects
                    inputs_seen[phase] = 0.0

        global_summary_step[phase] = summary_step
        epoch_loss = running_loss / sizes[phase]
        if problem == 'classification' or problem == 'ordinal':
            epoch_acc = running_corrects / sizes[phase]

        loss_array[phase].append(epoch_loss)
        if problem == 'classification' or problem == 'ordinal':
            acc_array[phase].append(epoch_acc)

        if verbose:
            if problem == 'classification' or problem == 'ordinal':
                print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                        phase, epoch_loss, epoch_acc))
            else:
                print('{} Loss: {:.4f}'.format(
                        phase, epoch_loss))
        
        if phase == 'val' and epoch_loss < best_loss:
            if problem == 'classification' or problem == 'ordinal':
                best_acc = epoch_acc
            best_loss = epoch_loss
            torch.save(model.state_dict(), os.path.join(save_dir, 'model_dict_best.pt'))
            best_epoch = epoch

    torch.save(model.state_dict(), os.path.join(save_dir, 'model_last.pt'))
    
    model.load_state_dict(torch.load(os.path.join(save_dir, 'model_dict_best.pt')))

    results = {
            'best_epoch': best_epoch,
            'best_loss': best_loss
        }

    if problem == 'classification' or problem == 'ordinal':
        results['best_acc'] =  best_acc

    return model, results

def evaluate_fusion(model, dataloader, dataset_size, transforms, criterion,
             device='cpu', problem='classification', verbose=True):
    """ 
    Evaluate classification/regression fusion model on test set
        Parameters:
            model (torch.nn.Module): Pytorch model already declared.
            dataloasder (torch.utils.data.DataLoader): dataloader with the dataset
            dataset_size (int): Size of the dataset.
            transforms (torch.nn.Sequential): Transforms to be applied to the data
            device (str): Device to move the data to. Default: cpu.
            problem (str): if it is a classification or regresion problem
            verbose (bool): wether or not to display metrics at the end

        Returns:
            test_results (dict): dictionary containing the labels, predictions,
                                 probabilities and accuracy of the model on the dataset.
    """
    model.eval()

    corrects = 0
    predictions = []
    probabilities = []
    real = []
    losses = []
    for batch in tqdm(dataloader):        
        wsi = batch[0]
        rna = batch[1]
        labels = batch[2]

        labels = labels.flatten()
        labels = labels.to(device)

        wsi = wsi.to(device)
        rna = rna.to(device)
        wsi = transforms(wsi)
        with torch.set_grad_enabled(False):
            outputs, embd1, embd2 = model(rna, wsi)
            if problem == 'classification':
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
            elif problem == 'regression':
                loss = criterion(outputs, labels.view(labels.size(0), 1))
            elif problem == 'ordinal':
                 _, preds = torch.max(outputs, 1)
                 loss = criterion(outputs, labels.view(labels.size(0), 1))
    
        if problem == 'classification' or problem == 'ordinal':
            predictions.append(preds.detach().to('cpu').numpy())
            corrects += torch.sum(preds == labels)
        probabilities.append(outputs.detach().to('cpu').numpy())
        real.append(labels.detach().to('cpu').numpy())
        losses.append(loss.detach().item())

    if problem == 'classification' or problem == 'ordinal':
        accuracy = corrects / dataset_size
    predictions = np.concatenate([predictions], axis=0, dtype=object)
    probabilities = np.concatenate([probabilities], axis=0, dtype=object)
    real = np.concatenate([real], axis=0, dtype=object)
    if (problem == 'classification' or problem == 'ordinal') and verbose:
        print('Accuracy of the model {}'.format(accuracy))
    else:
        print('Loss of the model {}'.format(np.mean(losses)))
    
    test_results = {
        'outputs': probabilities,
        'real': real
    }

    if problem == 'classification' or problem == 'ordinal':
        test_results['accuracy'] = accuracy.detach().to('cpu').numpy()
        test_results['predictions'] = predictions

    return test_results