import h5py
import numpy as np
h5file = h5py.File("test-dev0.1.h5")
scores = np.array(h5file['scores'])
print(scores)
