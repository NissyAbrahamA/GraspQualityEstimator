import numpy as np
data = np.load('//users//2//220269470//dev//graspqualityestimator_testing//Input_NN//debug//002_master_chef_can.npz')
keys = data.files
for key in keys:
    values = data[key]
    print(f"Key: {key}, Number of Values: {values}")
