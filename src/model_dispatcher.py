import encoders
import decoders

import config

import segmentation_models_pytorch as smp

MODELS = {
    'smp_unet_resnet34' : smp.Unet('resnet34', encoder_weights='imagenet', classes=config.CLASSES, activation='softmax')
}