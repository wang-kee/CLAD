import numpy as np
import os
import PIL
import torch
import torchvision

from PIL import Image
from torch.utils.data import Subset
from torchvision import datasets

# Inspired by https://github.com/tanimutomo/cifar10-c-eval


def load_txt(path: str) -> list:
    return [line.rstrip("\n") for line in open(path)]


corruptions = [
    'gaussian_noise', 'shot_noise', 'impulse_noise',
    'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur',
    'snow', 'frost', 'fog', 'brightness',
    'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression',
    'speckle_noise', 'gaussian_blur', 'spatter', 'saturate'
]


class CIFAR10C(datasets.VisionDataset):
    def __init__(
        self,
        root: str,
        name: str,
        severity: int,
        input_transform=None,
        target_transform=None,
    ):
        assert name in corruptions
        super(CIFAR10C, self).__init__(root, input_transform, target_transform)

        data_path = os.path.join(root, name + ".npy")
        target_path = os.path.join(root, "labels.npy")

        start_ind = (severity - 1) * 10000
        end_ind = (severity) * 10000
        self.data = np.load(data_path)[start_ind:end_ind]
        self.targets = np.load(target_path)[start_ind:end_ind]
        self.input_transform = input_transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        img, targets = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.input_transform is not None:
            img = self.input_transform(img)
        if self.target_transform is not None:
            targets = self.target_transform(targets)

        return img, targets

    def __len__(self):
        return len(self.data)


class ImageNetC(datasets.VisionDataset):
    def __init__(
        self,
        root: str,
        name: str,
        severity: int,
        input_transform=None,
        target_transform=None,
    ):
        assert name in corruptions
        super(ImageNetC, self).__init__(
            root, input_transform, target_transform)

        data_path = os.path.join(root, name + "/" + str(severity) + "/")
        testset = datasets.ImageFolder(data_path, input_transform,)
        self.testset = testset

    def dataset(self):
        return self.testset
