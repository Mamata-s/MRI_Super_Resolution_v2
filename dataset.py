import torch
import  cv2, os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import model_utility as ut
import copy

import random
# from torch.utils.data.sampler import Sampler


class MRIDataset(Dataset):
    def __init__(self, image_dir, label_dir,dir_dict=None,transform=None,test=False):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.images = os.listdir(image_dir)
        self.labels = os.listdir(label_dir)
        self.test = test
        self.dir_dict = dir_dict
        self.indices = [[] for _ in range(3)]
        for i, x in enumerate(self.images):
            img_path = os.path.join(self.image_dir, x)
            image = np.array(Image.open(img_path).convert('L'))  #to convert to grayscale
            if image.shape[0]== 256 and image.shape[1]== 152 : self.indices[0].append(i)
            if image.shape[0]== 360 and image.shape[1]== 152 : self.indices[1].append(i)
            if image.shape[0]== 360 and image.shape[1]== 256 : self.indices[2].append(i)

    def __len__(self):
        return len(self.images)

    def classes(self):
        return self.indices

    def __getitem__(self, index):
        dict_key = self.images[index]
        img_path = os.path.join(self.image_dir, self.images[index])
        label_path = os.path.join(self.label_dir, self.dir_dict[dict_key])
        image = np.array(Image.open(img_path).convert('L'))  #to convert to grayscale
        image = torch.from_numpy(image)
        label = np.array(Image.open(label_path).convert('L'))
        label = torch.from_numpy(label)

        if self.transform is not None:
            image = self.transform(image)
            label = self.transform(label)
        #normalize input and label image
        image = ut. min_max_normalize(image)
        label = ut.min_max_normalize(label)
        parser= (label - image)
        # parser= torch.abs(label - image) #change the parser from simply (label-image) to abs(label-image)
        image= torch.unsqueeze(image.float(),0)
        label = torch.unsqueeze(label.float(),0)
        parser = torch.unsqueeze(parser.float(),0)
        
        return image,label,parser


class RdnSampler():
    def __init__(self, data_source, batch_size, shuffle=True,classes=[]):
        self.classes = classes
        classes = copy.deepcopy(self.classes)
        self.indices = [[i for _ in range(len(klass))] for i, klass in enumerate(classes)]
        self.data_source = data_source
        self.batch_size = batch_size
        self.shuffle = shuffle

    def flatten_list(self, lst):
        return [item for sublist in lst for item in sublist]

    def __iter__(self):
        batch_lists = []
        for cluster_indices in self.indices:
            batches = [cluster_indices[i:i + self.batch_size] for i in range(0, len(cluster_indices), self.batch_size)]
            # filter our the shorter batches
            batches = [_ for _ in batches if len(_) == self.batch_size]
            if self.shuffle:
                random.shuffle(batches)
            batch_lists.append(batches)       
        
        # flatten lists and shuffle the batches if necessary
        # this works on batch level
        lst = self.flatten_list(batch_lists)
        if self.shuffle:
            random.shuffle(lst)
        # final flatten  - produce flat list of indexes
        lst = self.flatten_list(lst)        
        return iter(lst)

    def __len__(self):
        return len(self.data_source)


class MRIDatasetUpscale(Dataset):
    def __init__(self, image_dir, label_dir,dir_dict=None,transform=None,test=False):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.images = os.listdir(image_dir)
        self.labels = os.listdir(label_dir)
        self.test = test
        self.dir_dict = dir_dict
        self.indices = [[] for _ in range(3)]
        for i, x in enumerate(self.images):
            img_path = os.path.join(self.image_dir, x)
            image = np.array(Image.open(img_path).convert('L'))  #to convert to grayscale
            if image.shape[0]== 128 and image.shape[1]== 76 : self.indices[0].append(i)
            if image.shape[0]== 180 and image.shape[1]== 76 : self.indices[1].append(i)
            if image.shape[0]== 180 and image.shape[1]== 128 : self.indices[2].append(i)

    def __len__(self):
        return len(self.images)

    def classes(self):
        return self.indices

    def __getitem__(self, index):
        dict_key = self.images[index]
        img_path = os.path.join(self.image_dir, self.images[index])
        label_path = os.path.join(self.label_dir, self.dir_dict[dict_key])
        image = np.array(Image.open(img_path).convert('L'))  #to convert to grayscale
        image = torch.from_numpy(image)
        label = np.array(Image.open(label_path).convert('L'))
        label = torch.from_numpy(label)

        if self.transform is not None:
            image = self.transform(image)
            label = self.transform(label)
        #normalize input and label image
        image = ut. min_max_normalize(image)
        label = ut.min_max_normalize(label)
        parser= (label - image)
        # parser= torch.abs(label - image) #change the parser from simply (label-image) to abs(label-image)
        image= torch.unsqueeze(image.float(),0)
        label = torch.unsqueeze(label.float(),0)
        parser = torch.unsqueeze(parser.float(),0)
        
        return image,label,parser