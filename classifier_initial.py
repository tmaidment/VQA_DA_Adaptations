from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from PIL import Image
from torchvision import transforms
from sklearn.metrics import roc_auc_score, recall_score, precision_score, accuracy_score, confusion_matrix
from torch.utils.data import Dataset
from sklearn.model_selection import KFold
import glob
from tensorboardX import SummaryWriter
from trains import Task
from tqdm import tqdm
import argparse
from torchvision.datasets import ImageFolder

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(0)

class TransferClassifier(nn.Module):
    def __init__(self,num_classes=4):
        super(TransferClassifier,self).__init__()
        #resnet sizes: 18, 34, 50, 101, 152
        self.feature_extractor = torchvision.models.resnet50(pretrained=True)
        ## freeze the weights
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        self.feature_extractor.fc = nn.Linear(self.feature_extractor.fc.in_features, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.softmax(x)
        return x

def train_model(model, criterion, optimizer, scheduler, dataloaders, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_auc = 0.0
    best_acc = 0.0
    best_epoch = 0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            predz = []
            labelz = []
            outputz = []

            # Iterate over data.
            t = tqdm(iter(dataloaders[phase]), leave=False, total=len(dataloaders[phase]))
            for inputs, labels in t:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)

                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() # * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                for output in outputs.detach().cpu().numpy():
                    outputz.append(output[1])
                for pred in preds.cpu().numpy():
                    predz.append(pred)
                for label in labels.data.cpu().numpy():
                    labelz.append(label)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            #epoch_acc = accuracy_score(np.asarray(labelz), np.asarray(predz))
            y_true = np.asarray(labelz)
            y_pred = np.asarray(predz)
            y_conf = np.asarray(outputz)
            epoch_auc = roc_auc_score(y_true, y_conf)
            epoch_precision = precision_score(y_true, y_pred)
            epoch_recall = recall_score(y_true, y_pred)
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

            writer.add_scalar(phase+'/loss', epoch_loss, epoch)
            writer.add_scalar(phase+'/acc', epoch_acc, epoch)
            writer.add_scalar(phase+'/auc', epoch_auc, epoch)
            writer.add_scalar(phase+'/precision', epoch_precision, epoch)
            writer.add_scalar(phase+'/recall', epoch_recall, epoch)
            writer.add_scalar(phase+'/specificity', tn / (tn+fp), epoch)
            print('{} Loss: {:.4f} Acc: {:.4f} Auc:{:.4f}'.format(
                phase, epoch_loss, epoch_acc, epoch_auc))
            if phase != 'train':
                print('Labels: {}\nPreds: {}\nConfs: {}'.format(labelz, predz, outputz))

            if phase == 'test':
                if epoch_auc > best_auc:
                    best_auc = epoch_auc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    best_epoch = epoch
                elif epoch_auc == best_auc:
                    if epoch_acc > best_acc:
                        best_acc = epoch_acc
                        best_auc = epoch_auc
                        best_model_wts = copy.deepcopy(model.state_dict())
                        best_epoch = epoch
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, best_epoch

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a CNN for the VQA representation, via transfer learning.')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate float (default=0.001)')
    parser.add_argument('--gpu', default=0, type=int, help='gpu to train on (default=0)')
    parser.add_argument('--data_dir', type=str, help='the folder containing the preprocessed images')
    args = parser.parse_args()

    device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
    #device = torch.device('cpu')
    cpu_device = torch.device('cpu')

    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(256),
            transforms.RandomResizedCrop(224),
            #transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    model_conv = TransferClassifier()
    model_conv.to(device)

    run_string = "test1_lr{}".format(args.lr)
    writer = SummaryWriter(comment="_{}".format(run_string))
    image_datasets = {}
    #train dataset
    image_datasets['train'] = ImageFolder(os.path.join(args.data_dir, 'test'),transform=data_transforms['train'])
    #test dataset
    image_datasets['val'] = ImageFolder(os.path.join(args.data_dir, 'val'),transform=data_transforms['val'])

    print("Num Train: {} Num Val: {}".format(len(image_datasets['train']), len(image_datasets['val'])))

    # image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
    #                                         data_transforms[x])
    #                 for x in ['train', 'val']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=8,
                                                shuffle=x=='train', num_workers=4)
                for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

    weights = np.asarray([(np.asarray(image_datasets['train'].labels) == val).sum() for val in [0, 1, 2, 3]])
    weights = torch.tensor(weights.sum()/weights, dtype=torch.float).to(device)
    print('Class Weighting: {}'.format(weights))
    criterion = nn.CrossEntropyLoss(weight=weights)

    # Observe that only parameters of final layer are being optimized as
    # opposed to before.
    optimizer_conv = optim.SGD(model_conv.parameters(), lr=args.lr, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=25, gamma=0.1)

    model_conv, epoch = train_model(model_conv, criterion, optimizer_conv, exp_lr_scheduler, dataloaders, num_epochs=25)

    torch.save(model_conv.state_dict(), '{}.pth'.format(run_string))