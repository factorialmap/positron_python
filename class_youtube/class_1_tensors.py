import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

#create a vector
torch.arange(10)

#reshape tensor
torch.arange(30).reshape(2,5,3)


#general broadcasting rules
a = np.array([1,2])
b = np.array([3,4])

a*b
