CROP_SIZE = 1024
CHANNELS = 3
CLASSES = 2

BATCH_SIZE = 128

INPUT_SHAPE = (CHANNELS, CROP_SIZE, CROP_SIZE)


INPUT = 'input'

TRAIN_PATH = f'{INPUT}/train_hq'
MASK_PATH = f'{INPUT}/train_masks'
TEST_PATH = f'{INPUT}/test_hq'

TRAIN_CSV = f'{INPUT}/train_masks.csv'
TEST_CSV = f'{INPUT}/sample_submission.csv'

MODEL_MEAN = (0.485, 0.456, 0.406)
MODEL_STD = (0.229, 0.224, 0.225)