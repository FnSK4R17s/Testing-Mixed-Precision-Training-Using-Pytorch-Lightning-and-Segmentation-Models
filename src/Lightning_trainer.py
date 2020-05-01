import config
from Lightning_module import CarvanaModel

import pytorch_lightning as pl


train_folds = [0, 1, 2, 3]
val_folds = [4]

carvana_model = CarvanaModel(train_folds, val_folds)

# most basic trainer, uses good defaults (1 gpu)
trainer = pl.Trainer(gpus=1)    
trainer.fit(carvana_model)   

