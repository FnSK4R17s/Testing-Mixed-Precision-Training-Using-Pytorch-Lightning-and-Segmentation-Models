import pandas as pd
import config
import os

from sklearn.model_selection import KFold


df = pd.read_csv(config.TRAIN_CSV)
df.loc[:,'kfold'] = -1



format = lambda x: os.path.splitext(os.path.basename(x))[0]

df['img'] = df.img.map(format)


kf = KFold(n_splits=5, random_state=42, shuffle=True) 
folds = kf.split(df.img.values)

for fold, (x, y) in enumerate(folds):
    print(len(y), fold)
    df.loc[y, 'kfold'] = fold

print(df.kfold.value_counts())
df.to_csv(config.TRAIN_FOLDS, index=False)

# print(df.head(10))