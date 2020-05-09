CROP_SIZE = 1024
CHANNELS = 3
CLASSES = 1

EPOCHS = 7

LR = 0.001

INPUT_SHAPE = (CHANNELS, CROP_SIZE, CROP_SIZE)

MODEL_NAME = 'smp_unet_resnet34'


INPUT = 'input'
OUTPUT = 'output'

TRAIN_PATH = f'{INPUT}/train_hq'
MASK_PATH = f'{INPUT}/train_masks'
TEST_PATH = f'{INPUT}/test_hq'

TRAIN_CSV = f'{INPUT}/train_masks.csv'
TEST_CSV = f'{INPUT}/sample_submission.csv'

MODEL_MEAN = (0.485, 0.456, 0.406)
MODEL_STD = (0.229, 0.224, 0.225)

TRAIN_FOLDS = f'{OUTPUT}/train_folds.csv'

TRAIN_BATCH_SIZE = 4
VAL_BATCH_SIZE = 4


logs_path = 'lightning_logs'
version = 'version_7'
ckpt_name = 'epoch=4'

PATH = f'{logs_path}/{version}/checkpoints/{ckpt_name}.ckpt'