import os
import pickle
import random
import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import heapq
from collections import namedtuple, deque

NETWORK_PARAMS = 'params8'

ACTIONS = ['LEFT', 'RIGHT', 'UP', 'DOWN','WAIT','BOMB']
#ACTIONS = ['LEFT', 'RIGHT', 'UP', 'DOWN']
MOVES = np.array([[1,0],[0,1],[-1,0],[0,-1]])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class QNetwork(nn.Module):
    """ Actor (Policy) Model."""
    def __init__(self,action_size):
        """
        Initialize parameters and build model.
        Params
        =======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            fc1_unit (int): Number of nodes in first hidden layer
            fc2_unit (int): Number of nodes in second hidden layer
        """
        super(QNetwork,self).__init__() ## calls __init__ method of nn.Module class
        self.action_size = action_size
        self.conv1 = nn.Conv2d(in_channels=2, out_channels=32, kernel_size=4, stride=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=2, stride=1)
        self.fc1 = nn.Linear(64 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, self.action_size)
        
    def forward(self,x):
        # x = state
        """
        Build a network that maps state -> action values.
        """
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # Flattening
        x = x.flatten()
        print(x)
        #print(np.shape(x.numpy()))
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def setup(self):

    #Q- Network
    self.qnetwork = QNetwork(len(ACTIONS)).to(device)
    self.input_shape = (2,17,17)
    #self.qnetwork_target = QNetwork(4, len(ACTIONS)).to(device)

    #for name, param in self.qnetwork_local.named_parameters():
    #    if param.requires_grad:
    #        print(name, param.data)

    if self.train:
        self.logger.info("Trainiere ein neues Model.")

    else:
        #self.logger.info(f"Lade Model '{PARAMETERS}'.")
        filename = os.path.join("parameters", f'{NETWORK_PARAMS}.pt')
        self.qnetwork.load_state_dict(torch.load(filename))
        print("params loaded")
        #for name, param in self.qnetwork_local.named_parameters():
        #    if param.requires_grad:
        #        print(name, param.data)
        #self.qnetwork_local.eval()

def act(self, game_state: dict) -> str:
    """Returns action for given state as per current policy
    Params
    =======
        features (array_like): current features
        eps (float): epsilon, for epsilon-greedy action selection
    """
    features = state_to_features(self, game_state)
    self.qnetwork.eval()
    with torch.no_grad():
        action_values = self.qnetwork(features)
    #print(action_values)
    #Epsilon -greedy action selction
    if self.train:
        if random.random() > self.eps:
            bestAction = ACTIONS[np.argmax(action_values.cpu().data.numpy())]
            #print(action_values, bestAction)
            return bestAction
        else:
            #print("RANDOM")
            return np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .1, .1])
            #return np.random.choice(ACTIONS, p=[.25, .25, .25, .25])
    else:
        bestAction = ACTIONS[np.argmax(action_values.cpu().data.numpy())]
        #print(action_values, bestAction)
        return bestAction

def state_to_features(self, game_state: dict) -> np.array:
    
    channel1 = game_state['field']
    channel2 = game_state['explosion_map']

    feat = np.array([channel1,channel2])

    features = torch.from_numpy(feat).float()
    #print(feat)
    return features

