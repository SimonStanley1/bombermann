from collections import namedtuple, deque

import pickle
from typing import List

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt
import random

import events as e
from .callbacks import state_to_features
from .callbacks import act


#ACTIONS_IDX = {'LEFT':0, 'RIGHT':1, 'UP':2, 'DOWN':3, 'WAIT':4, 'BOMB':5}
ACTIONS_IDX = {'LEFT':0, 'RIGHT':1, 'UP':2, 'DOWN':3}

BUFFER_SIZE = int(1e5)  #replay buffer size
BATCH_SIZE = 6         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-5               # learning rate
#UPDATE_EVERY = 20        # how often to update the network
TOTAL_EPISODES = 25000

EPSILON_START = 1
EPSILON_END = 0.01
EPSILON_DECAY = 0.9995

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def step(self, features, action, reward, next_features, done):
    # Save experience in replay memory
    self.memory.add(features, action, reward, next_features, done)

    # Learn every UPDATE_EVERY time steps.
    self.t_step = (self.t_step+1)% BATCH_SIZE
    if self.t_step == 0:
        # If enough samples are available in memory, get radom subset and learn
        experience = self.memory.sample()
        learn(self, experience, GAMMA)
        #print("learning")
        self.memory.memory.clear() 
    #print(len(self.memory))

def learn(self, experiences, gamma):
    """Update value parameters using given batch of experience tuples.
    Params
    =======
        experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples
        gamma (float): discount factor
    """
    features, actions, rewards, next_features, dones = experiences
    ## TODO: compute and minimize the loss
    criterion = torch.nn.MSELoss()
    # Local model is one which we need to train so it's in training mode
    self.qnetwork_local.train()
    # Target model is one with which we need to get our target so it's in evaluation mode
    # So that when we do a forward pass with target model it does not calculate gradient.
    # We will update target model weights with soft_update function
    self.qnetwork_target.eval()
    #shape of output from the model (batch_size,action_dim) = (64,6)
    predicted_targets = self.qnetwork_local(features).gather(1,actions)
    
    with torch.no_grad():
        labels_next = self.qnetwork_target(next_features).detach().max(1)[0].unsqueeze(1)

    # .detach() ->  Returns a new Tensor, detached from the current graph.
    labels = rewards + (gamma* labels_next*(1-dones))
        
    loss = criterion(predicted_targets,labels).to(device)
    self.loss.append(loss.item())
    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()

    # ------------------- update target network ------------------- #
    soft_update(self, self.qnetwork_local,self.qnetwork_target,TAU)
            
def soft_update(self, local_model, target_model, tau):
    """Soft update model parameters.
    θ_target = τ*θ_local + (1 - τ)*θ_target
    Params
    =======
        local model (PyTorch model): weights will be copied from
        target model (PyTorch model): weights will be copied to
        tau (float): interpolation parameter
    """
    for target_param, local_param in zip(target_model.parameters(),
                                       local_model.parameters()):
        target_param.data.copy_(tau*local_param.data + (1-tau)*target_param.data)
            
class ReplayBuffer:
    """Fixed -size buffe to store experience tuples."""
    
    def __init__(self, action_size, buffer_size, batch_size):
        """Initialize a ReplayBuffer object.
        
        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experiences = namedtuple("Experience", field_names=["state",
                                                               "action",
                                                               "reward",
                                                               "next_state",
                                                               "done"])
        
    def add(self,features, action, reward, next_features,done):
        """Add a new experience to memory."""
        e = self.experiences(features,action,reward,next_features,done)
        self.memory.append(e)
        
    def sample(self):
        """Randomly sample a batch of experiences from memory"""
        experiences = random.sample(self.memory,k=self.batch_size)
        
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([ACTIONS_IDX[e.action] for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        
        return (states,actions,rewards,next_states,dones)
    
    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

def setup_training(self):

    self.optimizer = optim.Adam(self.qnetwork_local.parameters(),lr=LR)
        
    # Replay memory 
    self.memory = ReplayBuffer(6, BUFFER_SIZE,BATCH_SIZE)
    # Initialize time step (for updating every UPDATE_EVERY steps)
    self.t_step = 0
    self.score = 0
    self.scores = []
    self.loss = []
    self.eps = EPSILON_START
    self.episode = 0


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    old_features = state_to_features(self, old_game_state)
    new_features = state_to_features(self, new_game_state)
    reward = reward_from_events(self, events)
    if new_game_state == None:
        done = 1
    else:
        done = 0
    
    step(self,old_features,self_action,reward,new_features,done)
    self.score += reward
    #print(self.score)
    

def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    self.scores.append(self.score)
    self.score = 0
    self.eps = max(self.eps*EPSILON_DECAY, EPSILON_END)
    self.episode += 1
    if self.episode >= TOTAL_EPISODES:
        #print(self.scores)
        x = np.arange(0,len(self.loss),1)
        #print(x)
        plt.plot(x,self.loss)
        plt.show()
        #for name, param in self.qnetwork_local.named_parameters():
        #    if param.requires_grad:
        #        print(name, param.data)
        torch.save(self.qnetwork_local.state_dict(), "parameters/params.pt")

    
def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        e.COIN_COLLECTED: 15,
        e.KILLED_OPPONENT: 5,
        e.MOVED_DOWN : -1,
        e.MOVED_LEFT : -1,
        e.MOVED_RIGHT : -1,
        e.MOVED_UP : -1,
        e.WAITED : -50,
        e.INVALID_ACTION : -10,
        e.KILLED_SELF : -100,
        e.SURVIVED_ROUND : 0
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum