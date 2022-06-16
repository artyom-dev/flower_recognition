import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
#import splitfolders
import os 
import torchvision 
from sklearn.model_selection import train_test_split 
from torch.utils.data import Dataset, DataLoader, random_split, SubsetRandomSampler, WeightedRandomSampler
from torchvision import datasets, transforms
import torch 
from pathlib import Path
import pathlib
import seaborn as sns 





def count_images(paths):
    ''' 
    Use for count files in each folder
    paths = directory
    '''
    count = []
    for clas in paths:
        c = len(os.listdir(clas))
        count.append(c)
    return count

def vizualize_ratios(labels, counts):
    '''Vizualization of ratios in percentage
        labels = List[classes]
        counts = List[counts]
    '''
    colors = sns.color_palette('pastel')[0:5]
    plt.figure(figsize = (10, 10))
    plt.pie(counts, labels = labels, colors = colors, autopct='%.0f%%')
    plt.show()



def read_dataset(root, size):
    
    '''
    Read, transform and create ImageFolder dataset:
    root = directory
    size = image size for trasnform
    '''
    flower_transform = transforms.Compose([transforms.Resize((size, size)),
                                       transforms.ToTensor(), 
                                       transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    flower_dataset = datasets.ImageFolder(root, transform=flower_transform)
    return flower_dataset 


def get_class_distribution(dataset_obj): 
    '''
    Create dict where key = class and value = count:
    
    dataset_obj = ImageFolder Dataset
    '''
    count_dict = {k:0 for k,v in dataset_obj.class_to_idx.items()}
    
    for element in dataset_obj:
        y_lbl = element[1]
        y_lbl = idx2class[y_lbl]
        count_dict[y_lbl] += 1
            
    return count_dict


def train_test_split_with_stratification(flower_dataset): 
    
    '''
    Divide dataset on train/test sets using sklearn func, with stratification:
    
    flower_dataset = ImageFolder Dataset
    '''
    
    targets = flower_dataset.targets

    train_idx, valid_idx= train_test_split(
        np.arange(len(targets)), test_size=0.2, random_state=42, shuffle=True, stratify=targets)
    
    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(valid_idx)
    
    train_loader = DataLoader(dataset=flower_dataset, shuffle=False, batch_size=64, sampler=train_sampler)
    val_loader = DataLoader(dataset=flower_dataset, shuffle=False, batch_size=64, sampler=val_sampler)
    
    return train_loader, val_loader



def get_class_distribution_loaders(dataloader_obj, dataset_obj):
    '''
    Distribution of classes in dataloaders:
    dataloader_obj = DataLoader
    dataset_obj = ImageFolder Dataset
    
    '''
    count_dict = {k:0 for k,v in dataset_obj.class_to_idx.items()}
    
    for _,j in dataloader_obj:
        y_idx = j.item()
        y_lbl = idx2class[y_idx]
        count_dict[str(y_lbl)] += 1
            
    return count_dict

