import os
import re
import random

import pandas as pd
import torch
import torch.utils.data
from PIL import Image

from datasets.data_augment import (
    PairCompose, 
    PairRandomCrop, 
    PairToTensor
)
    
    
class UHDMdataset:
    def __init__(self, config):
        self.config = config

    def get_loaders(self):

        train_dataset = UHDMMoireDataset(os.path.join(self.config.data.data_dir, self.config.data.train_dataset, 'train'),
                                          mode='train',
                                          patch_size=self.config.data.patch_size,
                                          csv_file=self.config.data.train_data_csv)
        
        val_dataset = UHDMMoireDataset(os.path.join(self.config.data.data_dir, self.config.data.val_dataset, 'test'),
                                        mode='test',
                                        patch_size=self.config.data.patch_size,
                                        csv_file=self.config.data.val_data_csv)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.config.training.batch_size,
                                                   shuffle=True, num_workers=self.config.data.num_workers,
                                                   pin_memory=True)
        
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False,
                                                 num_workers=self.config.data.num_workers,
                                                 pin_memory=True)

        return train_loader, val_loader
  

class UHDMMoireDataset(torch.utils.data.Dataset):
    
    def __init__(self, root_dir, csv_file, mode='train', patch_size=256):
        self.annotations = pd.read_csv(os.path.join(root_dir, csv_file))
        self.root_dir = root_dir 
        self.num_images = len(self.annotations) 
        self.mode = mode
        self.patch_size = patch_size
        if self.mode=='train':
            self.transforms = PairCompose([
                PairRandomCrop(self.patch_size),
                PairToTensor()
            ])
        else:
            self.transforms = PairCompose([
                PairToTensor()
            ])
            
    def __len__(self):
        return self.num_images

    def __getitem__(self, idx):
        gt_name = os.path.join(os.path.join(self.root_dir), self.annotations.iloc[idx, 0])
        moire_name = os.path.join(os.path.join(self.root_dir), self.annotations.iloc[idx, 1])
        gt_image = Image.open(gt_name).convert('RGB')
        moire_image = Image.open(moire_name).convert('RGB')
        moire_image, gt_image = self.transforms(moire_image, gt_image)
        image_id = self.annotations.iloc[idx, 0][:4]
        return torch.cat([moire_image, gt_image], dim=0), image_id


class FHDMIdataset:
    def __init__(self, config):
        self.config = config

    def get_loaders(self):

        train_dataset = FHDMIMoireDataset(os.path.join(self.config.data.data_dir, self.config.data.train_dataset, 'train'),
                                          mode='train',
                                          patch_size=self.config.data.patch_size,
                                          csv_file=self.config.data.train_data_csv)
        
        val_dataset = FHDMIMoireDataset(os.path.join(self.config.data.data_dir, self.config.data.val_dataset, 'test'),
                                        mode='test',
                                        patch_size=self.config.data.patch_size,
                                        csv_file=self.config.data.val_data_csv)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.config.training.batch_size,
                                                   shuffle=True, num_workers=self.config.data.num_workers,
                                                   pin_memory=True)
        
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False,
                                                 num_workers=self.config.data.num_workers,
                                                 pin_memory=True)

        return train_loader, val_loader


class FHDMIMoireDataset(torch.utils.data.Dataset):

    def __init__(self, root_dir, csv_file, mode='train', patch_size=256):
        self.annotations = pd.read_csv(os.path.join(root_dir, csv_file))
        self.root_dir = root_dir
        self.num_images = len(self.annotations) 
        self.mode = mode
        self.patch_size = patch_size
        if self.mode=='train':
            self.transforms = PairCompose([
                PairRandomCrop(self.patch_size),
                PairToTensor()
            ])
        else:
            self.transforms = PairCompose([
                PairToTensor()
            ])
            
    def __len__(self):
        return self.num_images

    def __getitem__(self, idx):
        gt_name = os.path.join(os.path.join(self.root_dir), self.annotations.iloc[idx, 0])
        moire_name = os.path.join(os.path.join(self.root_dir), self.annotations.iloc[idx, 1])
        gt_image = Image.open(gt_name).convert('RGB')
        moire_image = Image.open(moire_name).convert('RGB')
        moire_image, gt_image = self.transforms(moire_image, gt_image)
        image_id = self.annotations.iloc[idx, 0][-9:-4]
        return torch.cat([moire_image, gt_image], dim=0), image_id
    
    
class TIPdataset:
    def __init__(self, config):
        self.config = config

    def get_loaders(self):
        
        train_dataset = TIPMoireDataset(root_dir=os.path.join(self.config.data.data_dir, self.config.data.train_dataset, 'train'),
                                        csv_file=self.config.data.train_data_csv,
                                        mode='train',
                                        patch_size=self.config.data.patch_size,
                                        )
        
        val_dataset = TIPMoireDataset(root_dir=os.path.join(self.config.data.data_dir, self.config.data.val_dataset, 'test'),
                                        csv_file=self.config.data.val_data_csv,
                                        mode='test',
                                        patch_size=self.config.data.patch_size,
                                        )

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.config.training.batch_size,
                                                   shuffle=True, num_workers=self.config.data.num_workers,
                                                   pin_memory=True)
        
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False,
                                                 num_workers=self.config.data.num_workers,
                                                 pin_memory=True)

        return train_loader, val_loader
    

class TIPMoireDataset(torch.utils.data.Dataset):
    
    def __init__(self, root_dir, csv_file, mode='train', patch_size=256):
        self.annotations = pd.read_csv(os.path.join(root_dir, csv_file))
        self.root_dir = root_dir
        self.num_images = len(self.annotations) 
        self.mode = mode
        self.patch_size = patch_size
        self.transforms = PairCompose([
                PairToTensor()
            ])
            
    def __len__(self):
        return self.num_images

    def __getitem__(self, idx):
        gt_name = os.path.join(os.path.join(self.root_dir, 'target'), self.annotations.iloc[idx, 0])
        moire_name = os.path.join(os.path.join(self.root_dir, 'source'), self.annotations.iloc[idx, 1])
        gt_image = Image.open(gt_name).convert('RGB')
        moire_image = Image.open(moire_name).convert('RGB')
        w, h = gt_image.size
        
        if self.mode == 'train':
            i = random.randint(-6, 6)
            j = random.randint(-6, 6)
            gt_image = gt_image.crop((int(w / 6) + i, int(h / 6) + j, int(w * 5 / 6) + i, int(h * 5 / 6) + j))
            moire_image = moire_image.crop((int(w / 6) + i, int(h / 6) + j, int(w * 5 / 6) + i, int(h * 5 / 6) + j))
            gt_image = gt_image.resize((self.patch_size, self.patch_size), Image.BILINEAR)
            moire_image = moire_image.resize((self.patch_size, self.patch_size), Image.BILINEAR)
            
        elif self.mode == 'test':
            gt_image = gt_image.crop((int(w / 6), int(h / 6), int(w * 5 / 6), int(h * 5 / 6)))
            moire_image = moire_image.crop((int(w / 6), int(h / 6), int(w * 5 / 6), int(h * 5 / 6)))
            gt_image = gt_image.resize((self.patch_size, self.patch_size), Image.BILINEAR)
            moire_image = moire_image.resize((self.patch_size, self.patch_size), Image.BILINEAR)
        else:
            print('Unrecognized mode! Please select either "train" or "test"')
            raise NotImplementedError

        moire_image, gt_image = self.transforms(moire_image, gt_image)
        image_id = self.annotations.iloc[idx, 0][:-11]
        return torch.cat([moire_image, gt_image], dim=0), image_id
