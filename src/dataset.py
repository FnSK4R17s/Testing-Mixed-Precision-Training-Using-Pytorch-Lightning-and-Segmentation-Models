import albumentations as A
import numpy as np
import pandas as pd
# import cv2
from PIL import Image
import torch

import config

class CarvanaDataset:
    def __init__(self, folds):
        df = pd.read_csv(config.TRAIN_FOLDS)
        df = df[['img', 'kfold']]
        df = df[df.kfold.isin(folds)].reset_index(drop=True)
        self.image_ids = df.img.values

        if len(folds) == 1:
            self.aug = A.Compose([
                A.Resize(config.CROP_SIZE, config.CROP_SIZE, always_apply=True),
                A.Normalize(config.MODEL_MEAN, config.MODEL_STD, always_apply=True)
            ])
        else:
            self.aug = A.Compose([
                A.Resize(config.CROP_SIZE, config.CROP_SIZE, always_apply=True),
                A.ShiftScaleRotate(
                    shift_limit=0.0625,
                    scale_limit=0.1,
                    rotate_limit=15,
                    p=0.9),
                A.Normalize(config.MODEL_MEAN, config.MODEL_STD, always_apply=True)
            ])
        

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, item):
        img_name = self.image_ids[item]
        image = np.array(Image.open(f'{config.TRAIN_PATH}/{img_name}.jpg'))
        mask = np.array(Image.open(f'{config.MASK_PATH}/{img_name}_mask.gif'))

        augmented = self.aug(image=image, mask=mask)

        image = augmented['image']
        mask = augmented['mask']

        image = np.transpose(image, (2, 0, 1)).astype(np.float32)

        return {
            'image': torch.tensor(image, dtype=torch.float),
            'mask': torch.tensor(mask, dtype=torch.float)
        }



