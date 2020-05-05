import config
from Lightning_module import CarvanaModel

import pytorch_lightning as pl


from PIL import Image
import numpy as np
import albumentations as A
import torch

import matplotlib.pyplot as plt

x = f'{config.TEST_PATH}/0bdb8b1cba05_01.jpg'


image = np.array(Image.open(x))


aug = A.Compose([
    A.Resize(config.CROP_SIZE, config.CROP_SIZE, always_apply=True),
    A.Normalize(config.MODEL_MEAN, config.MODEL_STD, always_apply=True)
    ])

image = aug(image=image)
image = image['image']

plt.imshow(image)

image = np.transpose(image, (2, 0, 1)).astype(np.float32)

image = torch.tensor(image, dtype=torch.float)

print(image.size())

image.unsqueeze_(0)

print(image.size())

train_folds = [0, 1, 2, 3]
val_folds = [4]

carvana_model = CarvanaModel.load_from_checkpoint(checkpoint_path=config.PATH, train_folds=train_folds, val_folds=val_folds)
carvana_model.eval()
mask = carvana_model(image)

mask.squeeze_()

print(mask)

mask = torch.argmax(mask,dim=0)

print(mask)

plt.imshow(mask.cpu().detach().numpy(), alpha=0.5)
plt.show()