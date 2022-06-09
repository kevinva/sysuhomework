import torch
from torch.utils.data import random_split
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

import os
from PIL import Image

from configs import *



def make_data(root_dir_path):
    root_dir = root_dir_path
    train_info = {}
    test_info = {}
    val_info = {}
    class2idx = {}
    torch.manual_seed(666)
    for i, dir in enumerate(os.listdir(root_dir)):
        class_name = dir.split('.')[1]
        class2idx[class_name] = i

        for root, subdirs, fnames in os.walk(os.path.join(root_dir, dir)):
            filepaths = [os.path.join(root_dir, dir, fname) for fname in fnames]
            num_train = int(len(filepaths) * 0.8)
            num_other = len(filepaths) - num_train
            num_test = int(num_other * 0.5)
            num_val = num_other - num_test
            train_set, other_set = random_split(filepaths, lengths=[num_train, num_other])
            test_set, val_set = random_split(other_set, lengths=[num_test, num_val])
            # print(len(test_set))

            train_info[class_name] = list(train_set)
            test_info[class_name] = list(test_set)
            val_info[class_name] = list(val_set)

    return class2idx, train_info, test_info, val_info


class CubDataset(Dataset):
  
  def __init__(self, path_info, class2idx, transform=None):
    self.transform = transform
    self.path_info = path_info
    self.class2idx = class2idx
    self.data = self.make_dataset()
  
  def make_dataset(self):
    items = list()
    for class_name, file_paths in self.path_info.items():
      label = [0] * len(self.path_info)
      label[self.class2idx[class_name]] = 1
      items.extend([(path, label) for path in file_paths])
    return items

  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    if torch.is_tensor(idx):
      idx = idx.tolist()
    path, target = self.data[idx]
    image = Image.open(path).convert('RGB')

    if self.transform:
      image = self.transform(image)
    
    return image, torch.tensor(target, dtype=torch.float32)
    

