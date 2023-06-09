from sklearn.model_selection import train_test_split
import os
import numpy as np
import cv2 as cv
import torch
from torch.utils.data import Dataset
import albumentations as A


class My_Dataset(Dataset):
    def __init__(self, x_path, y_path, x_data, y_data, transforms=True):
        self.x_path = x_path
        self.y_path = y_path
        self.x_data = x_data
        self.y_data = y_data
        self.transforms = transforms
        self.labels = ['Unlabeled', 
                       'Building',
                       'Fence',
                       'Other',
                       'Pedestrian',
                       'Pole',
                       'Road line',
                       'Road',
                       'Sidewalk',
                       'Vegetation',
                       'Car',
                       'Wall',
                       'Traffic sign']
        
    def __len__(self):
        return len(self.x_data)
    

    def __getitem__(self, index):
        img = cv.imread(os.path.join(self.x_path, self.x_data[index]))
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        img = cv.resize(img, (128, 128), interpolation=cv.INTER_NEAREST)
        seg_img = cv.imread(os.path.join(self.y_path, self.y_data[index]))
        seg_img = cv.cvtColor(seg_img, cv.COLOR_BGR2RGB)
        seg_img = cv.resize(seg_img, (128, 128), interpolation=cv.INTER_NEAREST)
        seg_img = seg_img[:,:,0]
        if self.transforms:
            img, seg_img = augment(img, seg_img)

        masks = []
        for i in range(len(self.labels)):
            mask = np.where(seg_img==i, 1, 0)
            masks.append(mask)
        masks = np.asarray(masks)
        img = torch.as_tensor(img, dtype=torch.float32)
        img = img.permute(2, 0, 1)
        masks = torch.as_tensor(masks, dtype=torch.float32)
        
        return img/255, masks

    
def augment(x, y):
    aug = A.Compose([
           A.RandomBrightnessContrast(p=0.2),
           A.GaussNoise(p=0.2),
           A.RGBShift(p=0.1),
           A.Spatter(p=0.1),
           A.HorizontalFlip(p=0.15),
           A.Rotate(p=0.2, limit=20, interpolation=cv.INTER_NEAREST),
        ])
    augmented = aug(image=x, mask=y)
    aug_img = augmented['image']
    aug_mask = augmented['mask']
    
    return aug_img, aug_mask


def make_datasets(x_path, y_path, test_only=False):
    x = os.listdir(x_path)
    y = os.listdir(y_path)
    
    X_train, X_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=1, shuffle=True)
    X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size=0.5, random_state=3, shuffle=True)
    
    train_dataset = My_Dataset(x_path, y_path, X_train, y_train, transforms=True)
    val_dataset = My_Dataset(x_path, y_path, X_val, y_val, transforms=True)
    test_dataset = My_Dataset(x_path, y_path, X_test, y_test, transforms=False)
    if test_only:
        return test_dataset

    return train_dataset, val_dataset
