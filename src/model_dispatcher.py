import encoders
import decoders

import config

import segmentation_models_pytorch as smp

MODELS = {
    'smp_unet_resnet34' : smp.Unet('resnet34', encoder_weights='imagenet', classes=config.CLASSES, activation='softmax'),
    'smp_unet_resnet50' : smp.Unet('resnet50', encoder_weights='imagenet', classes=config.CLASSES, activation='softmax'),
    'smp_unet_resnext50_32x4d' : smp.Unet('resnext50_32x4d', encoder_weights='imagenet', classes=config.CLASSES, activation='softmax'),
    'smp_unet_se_resnext50_32x4d' : smp.Unet('se_resnext50_32x4d', encoder_weights='imagenet', classes=config.CLASSES, activation='softmax'),
    'smp_unet_efficientnet-b4' : smp.Unet('efficientnet-b4', encoder_weights='imagenet', classes=config.CLASSES, activation='softmax'),
    'smp_unet_efficientnet-b7' : smp.Unet('efficientnet-b7', encoder_weights='imagenet', classes=config.CLASSES, activation='softmax'),
    'smp_unet_timm-efficientnet-b5' : smp.Unet('timm-efficientnet-b5', encoder_weights='imagenet', classes=config.CLASSES, activation='softmax')
}