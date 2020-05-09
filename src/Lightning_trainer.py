import config
from Lightning_module import CarvanaModel

import pytorch_lightning as pl


def train_iterative(train_folds, val_folds):
    carvana_model = CarvanaModel(train_folds, val_folds)

    # most basic trainer, uses good defaults (1 gpu)
    trainer = pl.Trainer(gpus=1, accumulate_grad_batches=64, amp_level='O1', precision=16, profiler=True, max_epochs=config.EPOCHS)
    trainer.fit(carvana_model)

    del trainer
    del carvana_model  

if __name__ == "__main__":

    train_iterative([4, 1, 2, 3],[0]) # 02:00:00
    train_iterative([0, 4, 2, 3],[1]) # 04:00:00
    train_iterative([0, 1, 4, 3],[2]) # 06:00:00
    train_iterative([0, 1, 2, 4],[3]) # 08:00:00
    train_iterative([0, 1, 2, 3],[4]) # 10:00:00