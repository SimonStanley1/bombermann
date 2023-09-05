import os
import pickle
import random
import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np

NETWORK_PARAMS = 'params'

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

class DeepQNet(nn.Module):
    def __init__(self, lr, input_dim, fc1_dim, fc2_dim, n_actions):
        super(DeepQNet, self).__init__()
        self.input_dim = input_dim
        self.fc1_dim = fc1_dim
        self.fc2_dim = fc2_dim
        self.n_actions = n_actions

        self.layer1 = nn.Linear(self.input_dim, self.fc1_dim)
        self.layer2 = nn.Linear(self.fc1_dim, self.fc2_dim)
        self.layer3 = nn.Linear(self.fc2_dim, self.n_actions)

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)
    
    def initialize_training(self, lr):
        self.loss = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    
    def forward(self, features):
        x = F.relu(self.layer1(features))
        x = F.relu(self.layer2(x))
        actions = self.layer3(x)

        return actions


def setup(self):
    """
    Setup your code. This is called once when loading each agent.
    Make sure that you prepare everything such that act(...) can be called.

    When in training mode, the separate `setup_training` in train.py is called
    after this method. This separation allows you to share your trained agent
    with other students, without revealing your training code.

    In this example, our model is a set of probabilities over actions
    that are is independent of the game state.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    self.network = DeepQNet(0.001, 4, 256, 256, 6)

    if self.train:
        self.logger.info("Trainiere ein neues Model.")

    else:
        #self.logger.info(f"Lade Model '{PARAMETERS}'.")
        filename = os.path.join("parameters", f'{NETWORK_PARAMS}.pt')
        self.network.load_state_dict(T.load(filename))
        self.network.eval()

    #if self.train or not os.path.isfile("my-saved-model.pt"):
    #    self.logger.info("Setting up model from scratch.")
    #    weights = np.random.rand(len(ACTIONS))
    #    self.model = weights / weights.sum()
    #else:
    #    self.logger.info("Loading model from saved state.")
    #   with open("my-saved-model.pt", "rb") as file:
    #       self.model = pickle.load(file)


def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    # todo Exploration vs exploitation
    random_prob = .5
    if self.train and random.random() < random_prob:
        self.logger.debug("Choosing action purely at random.")
        # 80%: walk in any direction. 10% wait. 10% bomb.
        return np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .1, .1])

    self.logger.debug("Querying model for action.")
    #return np.random.choice(ACTIONS, p=self.model)

    features = state_to_features(self, game_state)
    Q = self.network(features)

    print(Q)

    action_prob	= np.array(T.softmax(Q,dim=1).detach().squeeze())
    best_action = ACTIONS[np.argmax(action_prob)] 
    print(best_action)
    return best_action

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

    features = []
    # agents position 
    Xagent = game_state['self'][3][0]
    Yagent = game_state['self'][3][1]
    features.append(Xagent)
    features.append(Yagent)

    #find nearest coin
    dist_to_coins = np.sum((np.array(game_state['coins'])-np.array([Xagent,Yagent]))**2,axis=-1)
    nearest_coin_index = np.argmin(dist_to_coins)
    nearest_coin = game_state['coins'][nearest_coin_index]

    features.append(nearest_coin[0])
    features.append(nearest_coin[1])

    features = np.array(features)
    features = T.from_numpy(features).float()

    return features.unsqueeze(0)
