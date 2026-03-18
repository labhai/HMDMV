import os
import itertools

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

def train_transform(mean, std):
    transform = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomAffine(degrees=20, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
    return transform

def val_transform(mean, std):
    # Data augmentation for validation and test sets
    transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
    return transform

def generate_target2indices(targets):
    target2indices = {}
    for i in range(len(targets)):
        t = targets[i]
        if t not in target2indices:
            target2indices[t] = [i]
        else:
            target2indices[t].append(i)
    return target2indices

class HotelsDataset(Dataset):
    def __init__(self, data_dir, csv_file, n=2, train=False, classes=None):
        csv_path = os.path.join(data_dir, csv_file)
        self.data = pd.read_csv(csv_path)
        self.data['images'] = self.data['images'].apply(eval)
        self.image_dir = os.path.join(data_dir, 'images')

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        # Set transforms for train or validation mode
        if train:
            self.transform = train_transform(mean=mean, std=std)
        else:
            self.transform = val_transform(mean=mean, std=std)
        
        if classes is None:
            self.classes, self.class_to_idx = self.find_classes()
        else:
            # Assure using same classes indcies as generated for training dataset
            self.classes = classes
            self.class_to_idx = {self.classes[i]: i for i in range(len(self.classes))}

        self.num_classes = len(self.class_to_idx)
        self.n = n  # Number of images to sample
        self.train = train

        self.samples = self.make_dataset()
        self.image_paths = [s[0] for s in self.samples]
        self.targets = [int(s[1]) for s in self.samples]
        self.targets2indices = generate_target2indices(self.targets)

        if not self.train:
            self.samples = self.get_all_collection_combos()
            self.image_paths = [s[0] for s in self.samples]
            self.targets = [int(s[1]) for s in self.samples]

    def find_classes(self):
        classes = set()
        for _, row in self.data.iterrows():
            hotel_id = row['ID']
            classes.add(hotel_id)
        
        classes = list(classes)
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}

        return classes, class_to_idx
    
    def make_dataset(self):
        samples = []
        for _, row in self.data.iterrows():
            hotel_id = row['ID']
            images = row['images']
            class_idx = self.class_to_idx[hotel_id]
            
            for img in images:
                full_img_path = os.path.join(self.image_dir, img)
                samples.append((full_img_path, class_idx))
        return samples

    def get_all_collection_combos(self):
        samples = []
        for t, indices in self.targets2indices.items():
            paths = [self.image_paths[i] for i in indices]
            # If the number of available images is smaller than n,
            # duplicate existing images to form an n-view input.
            if len(paths) < self.n:
                temp_paths = list(paths)
                while len(temp_paths) < self.n:
                    temp_paths.append(np.random.choice(temp_paths))
                samples.append([temp_paths, t])
            else:
                for subset in itertools.combinations(paths, self.n):
                    samples.append([subset, t])
                
        return samples
        
    def __getitem__(self, index):
        target = self.targets[index]
        hotel_id = self.classes[target]
        
        if self.train:  # select random images to go with it
            possible_choices = self.targets2indices[target]
            
            if len(possible_choices) <= self.n:
                paths = [self.image_paths[i] for i in possible_choices]
                unique_requirement = False
            else:
                paths = [self.image_paths[index]]
                unique_requirement = True

            while len(paths) < self.n:
                selection = np.random.choice(possible_choices)
                path = self.image_paths[selection]
                if path not in paths or not unique_requirement:
                    paths.append(path)
        else:
            paths = self.image_paths[index]

        target = torch.ones((self.n, )).long() * target
        images = torch.stack([self.transform(Image.open(p).convert('RGB')) for p in paths])
        
        return images, target, hotel_id, paths

    def __len__(self):
        return len(self.samples)