from PIL import Image
import os
import os.path
import torch.utils.data
import torchvision.transforms as transforms
import numpy as np
import torch


def default_image_loader(path):
    return Image.open(path).convert('RGB')


class TransformTwice:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, inp):
        out1 = self.transform(inp)
        out2 = self.transform(inp)
        return out1, out2


class SimpleImageLoader(torch.utils.data.Dataset):
    # compared to the baseline code for SimpleImageLoader, we have added parameter called k (default=2)
    # k is for number of augmentation of train data. -> to increase the train data
    def __init__(self, rootdir, split, ids=None, k=2, transform=None, loader=default_image_loader):
        if split == 'test':
            self.impath = os.path.join(rootdir, 'test_data')
            meta_file = os.path.join(self.impath, 'test_meta.txt')
        else:
            self.impath = os.path.join(rootdir, 'train/train_data')
            meta_file = os.path.join(rootdir, 'train/train_label')

        imnames = []
        imclasses = []

        with open(meta_file, 'r') as rf:
            for i, line in enumerate(rf):
                if i == 0:
                    continue
                instance_id, label, file_name = line.strip().split()
                if int(label) == -1 and (split != 'unlabel' and split != 'test'):
                    continue
                if int(label) != -1 and (split == 'unlabel' or split == 'test'):
                    continue
                if (ids is None) or (int(instance_id) in ids):
                    if os.path.exists(os.path.join(self.impath, file_name)):
                        imnames.append(file_name)
                        if split == 'train' or split == 'val':
                            imclasses.append(int(label))

        self.transform = transform
        self.TransformTwice = TransformTwice(transform)
        self.loader = loader
        self.split = split
        self.imnames = imnames
        self.imclasses = imclasses
        self.k = k  # save k to self.k to be used below

    def __getitem__(self, index):
        filename = self.imnames[index]
        img = self.loader(os.path.join(self.impath, filename))

        if self.split == 'test':
            if self.transform is not None:
                img = self.transform(img)
            return img
        elif self.split == 'val':
            if self.transform is not None:
                img = self.transform(img)
            label = self.imclasses[index]
            return img, label
        elif self.split == 'train':
            for i in range(self.k):
                # the k is the no.aug introduced above, and originally comes from main_MT_TSA_transform.py
                # if k =2 for example, we will transform/give_augmentation to the training data
                # by k times so if train data were 1000 initially, we will end up with 2000 data.
                img1, img2 = self.TransformTwice(img)  # [3, 224, 224]
                imgsize = img1.size()
                img1rs = torch.reshape(img1, (1, imgsize[0], imgsize[1], imgsize[2]))
                img2rs = torch.reshape(img1, (1, imgsize[0], imgsize[1], imgsize[2]))
                if i == 0:
                    img1cat = img1rs
                    img2cat = img2rs
                else:
                    img1cat = torch.cat((img1cat, img1rs), 0)  # [20, 3, 224, 224]
                    img2cat = torch.cat((img2cat, img2rs), 0)
            # print("\timg SIZE = \t", img1.size())
            # print("\timgcat SIZE = \t", img1cat.size())
            label = self.imclasses[index]
            labelcat = torch.tensor([label] * self.k)  # [20]
            return img1cat, img2cat, labelcat
        else:  # unlabeled
            img1, img2 = self.TransformTwice(img)
            return img1, img2

    def __len__(self):
        return len(self.imnames)
