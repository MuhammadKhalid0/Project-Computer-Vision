import numpy as np

array = np.load('IMG_9939.npy')
print('Loaded array of size', array.shape)
print(array[:5,:5])
