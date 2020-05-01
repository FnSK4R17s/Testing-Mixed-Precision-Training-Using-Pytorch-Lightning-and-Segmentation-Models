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

plt.figure(1)
plt.subplot(211)
plt.imshow(np.transpose(img, (1, 2, 0)))

plt.subplot(212)
plt.imshow(mask)
plt.show()