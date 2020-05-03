import dataset
import matplotlib.pyplot as plt
import numpy as np

data = dataset.CarvanaDataset(folds=[1])

print(len(data))

idx = 24

img = data[idx]['image'].numpy()
mask = data[idx]['mask'].numpy()

print(img.shape)
print(mask.shape)


plt.imshow(np.transpose(img, (1, 2, 0)))

plt.imshow(mask, alpha=0.5)
plt.show()