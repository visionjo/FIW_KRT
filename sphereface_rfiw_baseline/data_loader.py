from torch.utils import data
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
from PIL import Image
import torch
import os
import random
import csv


class FIW_Train(data.Dataset):
    """Dataset class for FIW Training Set"""

    def __init__(self, root_dir, labels_path, n_classes, transform):
        self.root_dir = root_dir 
        self.labels_path = labels_path
        self.n_classes = n_classes
        self.transform = transform
        self.train_dataset = []
        self.preprocess()

    def preprocess(self):
        """Process the labels file"""
        lines = [line.rstrip() for line in open(self.labels_path, 'r')]

        for l in lines:
            spt = l.split()

            fname = spt[0]
            label = int(spt[1])
            #label_vec = [1 if i == label else 0 for i in range(self.n_classes)]

            self.train_dataset.append([fname, label])

    def __getitem__(self, index):
        """Return an image"""
        filename, label = self.train_dataset[index]
        image = Image.open(os.path.join(self.root_dir, filename))
        return self.transform(image), label

    def __len__(self):
        """Return the number of images."""
        return len(self.train_dataset)

class FIW_Val(data.Dataset):
    """Dataset class for FIW Validation Set"""

    def __init__(self, base_dir, csv_path, transform):
        self.base_dir = base_dir 
        self.csv_path = csv_path
        self.transform = transform
        self.val_dataset = []
        self.preprocess()

    def preprocess(self):
        """Process the pair CSVs"""
        with open(self.csv_path, 'r') as f:
            re = csv.reader(f)
            lines = list(re)

        self.val_dataset = [(l[2], l[3], bool(int(l[1]))) for l in lines]

    def __getitem__(self, index):
        """Return a pair"""
        path_a, path_b, label = self.val_dataset[index]
        img_a = self.transform(Image.open(os.path.join(self.base_dir, path_a)))
        img_b = self.transform(Image.open(os.path.join(self.base_dir, path_b)))
        return (img_a, img_b), label

    def __len__(self):
        """Return the number of images."""
        return len(self.val_dataset)

def get_train_loader(image_dir, labels_path='train/train.label', n_classes=300, image_size=(112, 96), batch_size=16, num_workers=1):
    """Build and return a data loader for the training set."""
    transform = []

    # Only used in training
    transform.append(T.RandomHorizontalFlip())

    transform.append(T.Resize(image_size))
    transform.append(T.ToTensor())
    transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    transform = T.Compose(transform)

    dataset = FIW_Train(image_dir, labels_path, n_classes, transform)

    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_workers)
    return data_loader

def get_val_loader(base_dir, csv_path, image_size=(112, 96), batch_size=128, num_workers=1):
    """Build and return a data loader for a split in the validation set."""
    transform = []

    transform.append(T.Resize(image_size))
    transform.append(T.ToTensor())
    transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    transform = T.Compose(transform)

    dataset = FIW_Val(base_dir, csv_path, transform)

    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  num_workers=num_workers)
    return data_loader
