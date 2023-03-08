import torch
import torchvision
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F

class parallel_dataloader(Dataset):
    """ 
    create a pairwise parralle dataloader for 3 datasets, s.t. that can be loaded and shuffled in same order 
    e.g., an anchor image can always be loaded in pair with its pre-segmented foreground.
    """
    
    def __init__(self, data_1, data_2, data_3):
        self.data_1 = data_1
        self.data_2 = data_2
        self.data_3 = data_3

    def __getitem__(self, index):
        xA = self.data_1[index]
        xB = self.data_2[index]
        xC = self.data_3[index]
        return xA, xB, xC

    def __len__(self):
        return len(self.data_1)    

def load_datasets(dir_data_train='datasets/original/train',
                  dir_data_train_fg='datasets/fg_grabcut/train',
                  dir_data_train_bg='datasets/only_bg_t/train',
                  dir_data_val='datasets/original/val',
                  dir_data_val_randbg='datasets/mixed_rand/val',
                  dir_data_val_samebg='datasets/mixed_same/val'):
    """ 
    Load respective datasets from disk. 
        data_train: original train set 
        data_train_fg: pre-segmented foreground of the train set
        data_train_bg: pre-segmented background of the train set
        data_val: original validation set
        data_val_randbg: random-background validation set
        data_val_samebg: random-background (from the same class) validation set
    """
    
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    transform_norm = transforms.Compose([
        transforms.RandomResizedCrop((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(),
        transforms.ToTensor(),
        normalize
    ])

    transform_fg = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor()
    ])

    transform_bg = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor()
    ])

    transform_val = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        normalize
    ])

    # training data
    # original images
    data_train = datasets.ImageFolder(root=dir_data_train, transform=transform_norm, target_transform=None)
    # fg images
    data_train_fg = datasets.ImageFolder(root=dir_data_train_fg, transform=transform_fg, target_transform=None)
    # bg images
    data_train_bg = datasets.ImageFolder(root=dir_data_train_bg, transform=transform_bg, target_transform=None)

    # validation data
    # original images
    data_val = datasets.ImageFolder(root=dir_data_val, transform=transform_val, target_transform=None)
    # randbg images
    data_val_randbg = datasets.ImageFolder(root=dir_data_val_randbg, transform=transform_val, target_transform=None)
    # samebg images
    data_val_samebg = datasets.ImageFolder(root=dir_data_val_samebg, transform=transform_val, target_transform=None)

    return data_train, data_train_fg, data_train_bg, data_val, data_val_randbg, data_val_samebg

def load_testsets(dir_data_test = 'datasets/test_set/original/val',
                  dir_data_test_fg = 'datasets/test_set/only_fg/val',
                  dir_data_test_randbg = 'datasets/test_set/mixed_rand/val',
                  dir_data_test_samebg = 'datasets/test_set/mixed_same/val',
                  dir_data_test_bg = 'datasets/test_set/only_bg_t/val',
                  batch_size=10):
    """ 
    Load respective test datasets from disk. 
        data_test: original test set 
        data_test_fg: only-foreground test set
        data_test_randbg: random-background test set
        data_test_samebg: random-background (from same class) test set
        data_test_bg: only-background test set
    """
    
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform_ = transforms.Compose([transforms.Resize(224), 
                                     transforms.CenterCrop((224, 224)), 
                                     transforms.ToTensor(), 
                                     normalize])
    
    # original data (with background)
    data_test =  datasets.ImageFolder(root = dir_data_test, transform = transform_, target_transform = None)
    # foreground data (without background)
    data_test_fg =  datasets.ImageFolder(root = dir_data_test_fg, transform = transform_, target_transform = None)
    # random background
    data_test_randbg =  datasets.ImageFolder(root = dir_data_test_randbg, transform = transform_, target_transform = None)
    # mixed same background
    data_test_samebg =  datasets.ImageFolder(root = dir_data_test_samebg, transform = transform_, target_transform = None)
    # only background _t 
    data_test_bg =  datasets.ImageFolder(root = dir_data_test_bg, transform = transform_, target_transform = None)
    
    # create data loaders respectively for each dataset
    test_loader = DataLoader(data_test, batch_size = batch_size, shuffle = False)
    test_loader_fg = DataLoader(data_test_fg, batch_size = batch_size, shuffle = False)
    test_loader_randbg = DataLoader(data_test_randbg, batch_size = batch_size, shuffle = False)
    test_loader_samebg = DataLoader(data_test_samebg, batch_size = batch_size, shuffle = False)
    test_loader_bg = DataLoader(data_test_bg, batch_size = batch_size, shuffle = False)
    
    return test_loader, test_loader_fg, test_loader_randbg, test_loader_samebg, test_loader_bg
