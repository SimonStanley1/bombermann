import os
import pickle
import random
import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np

NETWORK_PARAMS = 'params'

ACTIONS = ['LEFT', 'RIGHT', 'UP', 'DOWN','WAIT','BOMB']
#ACTIONS = ['LEFT', 'RIGHT', 'UP', 'DOWN']
MOVES = np.array([[1,0],[0,1],[-1,0],[0,-1]])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class QNetwork(nn.Module):
    """ Actor (Policy) Model."""
    def __init__(self, state_size,action_size, fc1_unit=256,
                 fc2_unit = 256):
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
    self.qnetwork = QNetwork(4, len(ACTIONS)).to(device)
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
    """
    *This is not a required function, but an idea to structure your code.*

    Converts the game state to the input of your model, i.e.
    a feature vector.

    You can find out about the state of the game environment via game_state,
    which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
    what it contains.

    :param game_state:  A dictionary describing the current game board.
    :return: np.array
    """
    # This is the dict before the game begins and after it ends
    #if game_state is None:
    #    return None
    #print(game_state['self'])
    # For example, you could construct several channels of equal shape, ...
    #channels = []
    #channels.append(...)
    # concatenate them as a feature tensor (they must have the same shape), ...
    #stacked_channels = np.stack(channels)
    # and return them as a vector

    if game_state is None:
        return None
    
    if not game_state['coins']:
        return torch.zeros(4)

    features = []
    # agents position 
    pos_agent = np.array(game_state['self'][3])

    #find nearest coin
    dist_to_coins = np.sum((np.array(game_state['coins'])-pos_agent)**2,axis=-1)
    nearest_coin_index = np.argmin(dist_to_coins)
    nearest_coin = game_state['coins'][nearest_coin_index]

    moves_to_coin = np.zeros(4)

    m = -1
    for move in pos_agent + MOVES:
       m += 1
       if game_state['field'][move[0],move[1]] == 0:
            dist_after_move = np.sqrt(np.sum((move-nearest_coin)**2))
            if dist_after_move == 0:
                inv_dist_after_move = 2
            else:
                inv_dist_after_move = 1/dist_after_move
            moves_to_coin[m] = inv_dist_after_move
    features = torch.from_numpy(moves_to_coin).float()
    #print(features)
    return features
