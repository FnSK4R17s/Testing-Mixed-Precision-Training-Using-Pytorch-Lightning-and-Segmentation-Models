import pytorch_lightning as pl
import model_dispatcher
import config
from dataset import CarvanaDataset
from dice_loss import IoULoss

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader


class CarvanaModel(pl.LightningModule):
    def __init__(self, train_folds, val_folds):
        super(CarvanaModel, self).__init__()
        # import model from model dispatcher
        self.model = model_dispatcher.MODELS['smp_unet_resnet34']
        self.train_folds = train_folds
        self.val_folds = val_folds

    def forward(self, x):
        # called with self(x)
        return self.model(x)

    def training_step(self, batch, batch_nb):
        # REQUIRED

        x = batch['image']
        y = batch['mask']
        # print(batch)
        y_hat = self(x)
        loss = IoULoss()(y_hat, y)
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_nb):
        # OPTIONAL
        x = batch['image']
        y = batch['mask']
        y_hat = self(x)
        return {'val_loss': IoULoss()(y_hat, y)}

    def validation_epoch_end(self, outputs):
        # OPTIONAL
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}
        return {'val_loss': avg_loss, 'log': tensorboard_logs}

    # def test_step(self, batch, batch_nb):
    #     # OPTIONAL
    #     x, y = batch
    #     y_hat = self(x)
    #     return {'test_loss': F.cross_entropy(y_hat, y)}

    # def test_epoch_end(self, outputs):
    #     # OPTIONAL
    #     avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
    #     logs = {'test_loss': avg_loss}
    #     return {'test_loss': avg_loss, 'log': logs, 'progress_bar': logs}

    def configure_optimizers(self):
        # REQUIRED
        # can return multiple optimizers and learning_rate schedulers
        # (LBFGS it is automatically supported, no need for closure function)
        return torch.optim.Adam(self.parameters(), lr=0.02)


    def train_dataloader(self):
        # REQUIRED
        return DataLoader(CarvanaDataset(folds=self.train_folds), batch_size=config.TRAIN_BATCH_SIZE)

    def val_dataloader(self):
        # OPTIONAL
        return DataLoader(CarvanaDataset(folds=self.val_folds), batch_size=config.VAL_BATCH_SIZE)

    # def test_dataloader(self):
    #     # OPTIONAL
    #     return DataLoader(MNIST(os.getcwd(), train=False, download=True, transform=transforms.ToTensor()), batch_size=32)