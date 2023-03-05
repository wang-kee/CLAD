import random
import torch
import torchvision
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F
import numpy as np

def set_seed(seed=42):
    """ Seed everything. """
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

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


class Neg_Sample_Dictionary():
    
    """ Dictionary to store negative samples for different label classes. """
    
    def __init__(self, target, latent, n_labels=9, n_neg_samples=25):
        """ Initialize the negative sample dictionary with trivial negative samples in the first iteration. """
        
        self.dic = {}
        self.n_labels = n_labels
        self.n_neg_samples = n_neg_samples
        
        target = target.cpu().numpy()
        latent = latent.clone().cpu().detach().numpy()
        for i in range(self.n_labels):
            neg_idx = np.where(target != i)[0]
            # if there're not enough negative samples, generate extra samples by replacement
            if len(neg_idx) < self.n_neg_samples:
                extra_replacement_idx = np.random.choice(neg_idx, size=self.n_neg_samples - len(neg_idx))
                neg_idx = np.append(neg_idx, extra_replacement_idx)
            self.dic[i] = latent[neg_idx[0:self.n_neg_samples]]
        
    def update_dict(self, bg_target, latent_pos, n_labels=9):
        """ Update the negative sample dictionary by putting new samples into respecitve keys of their background. """
        
        # generates N negative sample indexes for the target (without replacement)
        bg_target = bg_target.cpu().numpy()
        latent_pos = latent_pos.clone().cpu().detach().numpy()
        # dic stores the negative samples for the image with label i (same-class bg, different-class fg)
        for i in range(self.n_labels):
            # neg_idx_dic[i] represents the indexes for the samples in the batch with different labels
            bg_i_idx = np.where(bg_target == i)[0]
            if len(bg_i_idx) != 0:
                self.dic[i] = np.append(self.dic[i], latent_pos[bg_i_idx], axis=0)
                self.dic[i] = self.dic[i][len(bg_i_idx):]
        

def shuffle_fg_index(target):
    """ this function shuffles the index of the foreground, such that the original foreground corresponds to a foreground with different label """
    
    batch_size = len(target)
    
    target_ori = target.clone().detach()
    shuffle_index = np.arange(0, batch_size)
    # shuffle the index
    np.random.shuffle(shuffle_index)
    target_shuffle = target[shuffle_index]
    
    # fix situations where there still exists pairs which have the same foreground label
    for i in np.where(target_ori == target_shuffle)[0]:
        index_ori = shuffle_index[i].copy()
        while target_ori[shuffle_index[i]%batch_size] == target_ori[index_ori]:
            shuffle_index[i] += 1
        shuffle_index[i] = shuffle_index[i]%batch_size

    return shuffle_index

def cal_contrastive_loss(latent, latent_pos, target_copy, neg_sample_dic, args):
    
    """ Calculate the InfoNCE loss based on the latent representation of the anchor, positive samples and negative samples. """
    
    con_loss_denominator = 0

    positive_similarity = F.cosine_similarity(latent.float(), latent_pos.float())
    con_loss_nominator = torch.exp(positive_similarity / args.tau)
    con_loss_denominator += torch.exp(positive_similarity / args.tau)

    # pick negative samples from dictionary and calculate similarity
    for i in range(args.N_neg_samples):
        neg_samples = np.zeros((len(latent), latent.size(1)))
        for j in range(9):
            neg_samples[np.where(target_copy == j)] = neg_sample_dic[j][i]
        con_loss_denominator += torch.exp(F.cosine_similarity(latent.float(), torch.tensor(neg_samples).float().to(con_loss_nominator.device))/args.tau)

    contrastive_loss = -torch.log(con_loss_nominator / con_loss_denominator).mean()
    
    return contrastive_loss

def construct_pos_samples(fg, bg, augmentation=True):
    
    """ Construct positive samples by inpaiting the foreground on the randomly shuffled background. """
    
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    rand_transform_ = transforms.Compose([transforms.ToPILImage(),
                                          transforms.RandomResizedCrop((224, 224)),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ColorJitter(),
                                          transforms.ToTensor(),
                                          normalize])
    
    mask = 1 - torch.sign(torch.any(fg!=0, dim=1, keepdim=True).int())
    stacked = mask * bg + fg
    
    if augmentation:
        return torch.stack([rand_transform_(image) for image in stacked])
    else:
        return torch.stack([normalize(image) for image in stacked])

class parallel_dataset(Dataset):
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

class resnet_9l(nn.Module):
    """ adds intermediate layer output to pytorch resnet-50 structure (output changed to 9 classes) """
    
    def __init__(self):
        super(resnet_9l, self).__init__()
        
        resnet = torchvision.models.resnet50(pretrained = False)
        in_features = resnet.fc.in_features
        resnet.fc = nn.Linear(in_features, 9)
        
        self.resnet_layer = nn.Sequential(*list(resnet.children())[:-2])

        self.avgpool = list(resnet.children())[-2]
        self.fc = list(resnet.children())[-1]

    def forward(self, x):
        x = self.resnet_layer(x)

        latent_representation = self.avgpool(x).view(-1, 2048)

        logits = self.fc(latent_representation)
    
        return latent_representation, logits
    
def eval_model(model, 
               test_loader, 
               test_loader_fg, 
               test_loader_randbg, 
               test_loader_samebg, 
               test_loader_bg):
    
    """ 
    Evaluate the model on the following test set loaders:
        test_loader: original test set
        test_loader_fg: only-foreground test set
        test_loader_randbg: random-background test set
        test_loader_samebg: random-background (from same class) test set
        test_loader_bg: only-background test set
    """
    
    data_loader_names = ['original data', 'foreground data', 'random-background data', 'same-background data', 'only_bg data']
    
    model.eval()
    n_batch_val = int(4550 / 10)
    device = torch.device('cuda')
    
    print('Training complete, evaluting model on test set')
    for i, data_loader in enumerate([test_loader, test_loader_fg, test_loader_randbg, test_loader_samebg, test_loader_bg]):
        
        predictions = torch.tensor([]).to(device)
        targets = torch.tensor([]).to(device)

        with torch.no_grad():
            for b, (inputs, target) in enumerate(data_loader):
                inputs, target = inputs.to(device), target.to(device)
                inter, out = model(inputs)
                pred = out.max(dim=1)[1].to(device)
                predictions = torch.cat((predictions, pred), 0)
                targets = torch.cat((targets, target), 0)
                print("\rbatch: {:d} / {} ".format(b+1, n_batch_val), end='',  flush=True)

            accuracy = (predictions == targets).float().mean().item()
            print(' --- Test accuracy on {} for resnet: {:.4f}'.format(data_loader_names[i], accuracy))
