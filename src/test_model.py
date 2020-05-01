import model_dispatcher
import config

import torch
import torch.nn as nn

from torchsummary import summary

model = model_dispatcher.MODELS['smp_unet_resnet34']

model = model.to('cuda')

summary(model, config.INPUT_SHAPE)