import os
import numpy as np
from PIL import Image

import torch
import torch.utils.data as data
from torchvision import transforms


class classification_Datasets(data.Dataset):
    def __init__(self, data_list,input_size):
        super(classification_Datasets, self).__init__()

        # base setting
        self.data_list = data_list
        self.transform_img = transforms.Compose(
            [transforms.Resize((input_size,input_size)), transforms.RandomHorizontalFlip(0.5), transforms.ToTensor()
                , transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        # load data & label list
        img_paths = []
        img_labels = []
        trans_img=[]
        with open(self.data_list, 'r') as f:
            for line in f.readlines():
                string = line.split(' ')
                # img = Image.open(string[0])
                # img = self.transform_img(img).view(1,-1).numpy()
                # trans_img.append(img)
                img_paths.append(string[0])
                img_labels.append(int(string[1]))


        self.img_paths = img_paths
        self.img_labels = img_labels
        # self.trans_img=torch.from_numpy(img)

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        img = Image.open(img_path)
        img = self.transform_img(img)
        label = self.img_labels[index]
        label = torch.from_numpy(np.array(label))

        result = {
            'img': img,
            'label': label
        }

        return img,label

    def __len__(self):
        return len(self.img_labels)


class autoencoder_Datasets(data.Dataset):
    def __init__(self, data_list,input_size):
        super(classification_Datasets, self).__init__()

        # base setting
        self.data_list = data_list
        self.transform_img = transforms.Compose(
            [transforms.Resize((input_size,input_size)), transforms.RandomHorizontalFlip(0.5), transforms.ToTensor()
                , transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        # load data & label list
        img_paths = []
        img_labels = []

        with open(self.data_list, 'r') as f:
            for line in f.readlines():
                string = line.split(' ')
                img_paths.append(string[0])
                img_labels.append(int(string[1]))


        self.img_paths = img_paths
        self.img_labels = img_labels
        # self.trans_img=torch.from_numpy(img)

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        img = Image.open(img_path)
        img = self.transform_img(img)
        label = self.img_labels[index]
        label = torch.from_numpy(np.array(label))

        result = {
            'img': img,
            'label': label
        }

        return img,label

    def __len__(self):
        return len(self.img_labels)
