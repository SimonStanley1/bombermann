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

NETWORK_PARAMS = 'params'

ACTIONS = ['LEFT', 'RIGHT', 'UP', 'DOWN','WAIT','BOMB']
#ACTIONS = ['LEFT', 'RIGHT', 'UP', 'DOWN']
MOVES = np.array([[1,0],[0,1],[-1,0],[0,-1]])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class QNetwork(nn.Module):
    """ Actor (Policy) Model."""
    def __init__(self, state_size,action_size, fc1_unit=512,
                 fc2_unit = 512):
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
        self.state_size = state_size
        self.action_size = action_size
        self.fc1= nn.Linear(state_size,fc1_unit)
        self.fc2 = nn.Linear(fc1_unit,fc2_unit)
        self.fc3 = nn.Linear(fc2_unit,action_size)
        
    def forward(self,x):
        # x = state
        """
        Build a network that maps state -> action values.
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

def setup(self):

    #Q- Network
    self.qnetwork = QNetwork(1445, len(ACTIONS)).to(device)
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

    features = np.array([])

    field = game_state['field'].flatten()
    features = np.append(features, field)

    explosion_map = game_state['explosion_map'].flatten()
    features = np.append(features, explosion_map)

    pos_agent = np.array(game_state['self'][3])
    agents = np.zeros((17,17))
    agents[pos_agent[0],pos_agent[1]] = 1
    for oponent in game_state['others']:
        pos_oponent = oponent[3]
        agents[pos_oponent[0],pos_oponent[1]] = -1
    
    features = np.append(features, agents.flatten())

    coins = game_state['coins']
    coin_map = np.zeros((17,17))
    for coin in coins:
        coin_map[coin[0],coin[1]] = 1
    
    features = np.append(features, coin_map.flatten())

    danger = np.zeros((17,17))
    for bomb in game_state['bombs']:
        pos_bomb = bomb[0]
        bomb_timer = -1*(bomb[1]-1.5)+2.5
        danger[pos_bomb[0],pos_bomb[1]] = bomb_timer
        for move in MOVES:
            pos = pos_bomb
            for i in range(3):
                pos = pos + move
                field_value = game_state['field'][pos[0],pos[1]]
                if field_value == -1:
                    break
                danger[pos[0],pos[1]] = bomb_timer

    features = np.append(features, danger.flatten())
    features = torch.from_numpy(features).float()
    #print(features, len(features))

    return features

