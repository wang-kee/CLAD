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

def shuffle_fg_index(target):
    """ 
    Shuffle the index of the foreground, such that the shuffled copy has a different class label with the original copy for each sample of the batch.
    """
    
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