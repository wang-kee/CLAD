import random
import torch
import torchvision
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F
import numpy as np

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
