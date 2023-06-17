import torch
import numpy as np

probabilities = np.array([[np.log(1.),1.,np.log(5)]])
print(np.isnan(probabilities))
if True in np.isnan(probabilities):
	print("yes!")