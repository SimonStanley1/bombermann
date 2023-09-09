import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

import numpy as np
import matplotlib.pyplot as plt

network_params = 'params'
ACTIONS_IDX = {'LEFT':0, 'RIGHT':1, 'UP':2, 'DOWN':3, 'WAIT':4, 'BOMB':5}

t = torch.tensor([[1,2,3,4],[5,6,7,8]])
filename = os.path.join("parameters", f'{network_params}.pt')
#print(ACTIONS_IDX[['LEFT','UP']])

#torch.from_numpy(np.vstack(ACTIONS_IDX['LEFT','RIGHT','UP'])).long().to(device)
done = torch.tensor([1.0,1.0])
mask = (done == 1).nonzero(as_tuple=True)
#for i in range(3):
#    print(i)
#print(range(3))
t[mask] = 0.0
#print(t)

# = [[[1,1],3],[[2,2],2],[[3,3],1]]
b = (((1,1),3),((2,2),2))
#or i in b:
#    print(i[0])

#a = np.array([1,2])
#b = np.array([3,4])
print(np.zeros((17,17)))