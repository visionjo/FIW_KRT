from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from PIL import Image
import os
import csv


class FIW_DBBase(Dataset):
    def __init__(self, root_dir, labels_path, n_classes=300, transform=None):
        self.root_dir = root_dir
        self.labels_path = labels_path
        self.n_classes = n_classes
        self.transform = transform
        self.pairs = []

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pass



class FIW_Train(FIW_DBBase):
    """Dataset class for FIW Training Set"""

    def preprocess(self):
        """Process the labels file"""
        lines = [line.rstrip() for line in open(self.labels_path, 'r')]
        for l in lines:
            spt = l.split()
            fname = spt[0]
            label = int(spt[1])
            # label_vec = [1 if i == label else 0 for i in range(self.n_classes)]
            self.pairs.append([fname, label])

    def __getitem__(self, index):
        """Return an image"""
        filename, label = self.pairs[index]
        impath = os.path.join(self.root_dir, filename)
        image = Image.open(impath)
        return self.transform(image), label



class FIW_Val(FIW_DBBase):
    """Dataset class for FIW Validation Set"""

    def preprocess(self):
        """Process the pair CSVs"""
        with open(self.labels_path, 'r') as f:
            re = csv.reader(f)
            lines = list(re)

        self.pairs = [(l[2], l[3], bool(int(l[1]))) for l in lines]

    def __getitem__(self, index):
        """Return a pair"""
        path_a, path_b, label = self.pairs[index]
        img_a = self.transform(Image.open(os.path.join(self.root_dir, path_a)))
        img_b = self.transform(Image.open(os.path.join(self.root_dir, path_b)))
        return (img_a, img_b), label

    def __len__(self):
        """Return the number of images."""
        return len(self.pairs)





def get_train_loader(image_dir, labels_path='train/train.label', n_classes=300, image_size=(112, 96), batch_size=16,
                     num_workers=1):
    """Build and return a data loader for the training set."""
    transform = T.Compose([T.RandomHorizontalFlip(),
                           T.Resize(image_size),
                           T.ToTensor(),
                           T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
                           ])

    dataset = FIW_Train(image_dir, labels_path, n_classes=n_classes, transform=transform)
    dataset.preprocess()
    data_loader = DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_workers)
    return data_loader


def get_val_loader(base_dir, csv_path, image_size=(112, 96), batch_size=128, num_workers=1):
    """Build and return a data loader for a split in the validation set."""
    transform = T.Compose([T.Resize(image_size),
                           T.ToTensor(),
                           T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
                           ])

    dataset = FIW_Val(base_dir, csv_path, transform=transform)
    dataset.preprocess()
    data_loader = DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  num_workers=num_workers)
    return data_loader
