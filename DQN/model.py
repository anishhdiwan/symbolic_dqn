import random
from collections import namedtuple, deque
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class Environment:
	'''
	The environment class is used to create an env object for the first markov decision process that is used to generate trees using DQN.
	This is a simpler version of the OpenAI gym environment. It returns next states via deterministic transitions and does not return a 
	reward upon state transition. The reward is instead relayed to the agent from a second MDP chained up to this one. The second environment
	is the actual gym env. 
	'''

	def __init__(self, main_env):
        # main_env is an instance of the OpenAI gym environment (in this case, lunar lander)
		self.state = ExpressionTree()
		self.main_env = main_env
		self.done = False
        self.main_env_state = None

	def reset(self):
		self.state = ExpressionTree()
		return self.state

	def step(self, action):
        if not self.done:
    		self.state, tree_full = self.state.update(action)

            if tree_full:
                self.done = True

            state_eval = self.state.evaluate(self.main_env_state)
    		main_env_action = select_action(state_eval)
    		reward, done = main_env.step(main_env_action)
    		self.done = done
    		return self.state, reward, self.done
        else:
            print ("Episode complete")




# Setting up a transition for the replay memory
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state'))

# Defining the replay memory class for the agent's self-explored transitions
class ReplayMemory:
    '''
    Replay memory contains states and next_states that are stacked with FRAME_STACK number of observations. These are numpy arrays and not tensors.
    They are converted to tensors during optimization
    '''

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def append(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        # samle returns a list of transitions with batch_size number of elements
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

    def __iadd__(self, other): #changes and returns the existing memory
      self.memory += other.memory
      return self 

    def __add__(self, other): #leaves existing memory but creates and returns new combined memory
      self.memory = self.memory + other.memory 
      return self


# Defining the model class
class DQN(nn.Module):

    def __init__(self, n_observation_feats, n_actions, batch_size):
        super(DQfD, self).__init__()
        self.layer1 = nn.Linear(n_observation_feats, 32)
        self.layer2 = nn.Linear(32, 64)
        self.layer3 = nn.Linear(64, n_actions)
        self.batch_size = batch_size

    # Called with either one element to determine next action, or a batch
    def forward(self, x, for_optimization=True):
        # When using for optimization, a batch of inputs is passed in. In this case, reshape. When using for selecting actions, only one state is 
        # passed. In this case, the shape is already correctly set. Hence no reshaping is needed.
        if for_optimization: 
            x = torch.reshape(x, (self.batch_size,-1))
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)



# Defining the temporal difference loss function
class DQN_Loss(nn.Module):
    def __init__(self):
        super(DQN_Loss, self).__init__()


    def forward(self, policy_net, target_net, states, actions, rewards, next_states, GAMMA):
        # 1-step TD loss
        with torch.no_grad():
            next_state_max = torch.max(target_net(next_states), dim=1).values

        targets = rewards + GAMMA * next_state_max # torch.max(target_net(next_states), dim=1).values
        values = policy_net(states).gather(1,actions.view(-1,1)).view(-1,)

        return F.mse_loss(values, targets)


# Defining epsilon greedy action selection
def select_action(state, EPS, policy_net):
    state = state.to(device)
    sample = random.random()
    if sample > EPS:
        # print("Exploiting")
        with torch.no_grad():
            return torch.argmax(policy_net(state, for_optimization=False), dim=1).item()
    else:
        # print("Exploring")
        return action_names[random.choice(list(action_names.keys()))]
        # return action_list[action_names[random.choice(list(action_names.keys()))]]


# Defining the optimization for the Q-network
def optimize_model(optimizer, policy_net, target_net, replay_memory, dqn_loss, BATCH_SIZE = 32, GAMMA=0.99):
    '''
    Optimize the Q-network either using the agent's self-explored replay memory or using demo data. 
    The variable BETA defines the probability of sampling from either one. This will later be replaced by some importance sampling factor
    '''

    # print("Sampling from agent's replay memory")
    batch_transitions = replay_memory.sample(BATCH_SIZE)

    batch_states = []
    batch_actions = []
    batch_rewards = []
    batch_next_states = []
    batch_dones = []

    for i in range(BATCH_SIZE):
        batch_states.append(batch_transitions[i].state)
        batch_next_states.append(batch_transitions[i].next_state)
        batch_rewards.append(batch_transitions[i].reward)
        batch_actions.append(batch_transitions[i].action)

    # batch_states = torch.reshape(torch.tensor(np.array(batch_states), dtype=torch.float32, requires_grad=True), (BATCH_SIZE,-1))
    batch_states = torch.tensor(np.array(batch_states), dtype=torch.float32, requires_grad=True)
    # batch_next_states = torch.reshape(torch.tensor(np.array(batch_next_states), dtype=torch.float32, requires_grad=True), (BATCH_SIZE,-1))
    batch_next_states = torch.tensor(np.array(batch_next_states), dtype=torch.float32, requires_grad=True)
    batch_actions = torch.tensor(np.array(batch_actions))
    batch_rewards = torch.tensor(np.array(batch_rewards), dtype=torch.float32, requires_grad=True)
    batch_dones = torch.tensor(np.array(batch_dones))

    # print(batch_states.shape)
    # print(batch_next_states.shape)
    # print(batch_actions.shape)
    # print(batch_rewards.shape)
    # print(batch_dones.shape)
    batch_states, batch_next_states, batch_rewards, batch_actions = batch_states.to(device), batch_next_states.to(device), batch_rewards.to(device), batch_actions.to(device)
    loss = dqn_loss(policy_net, target_net, batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones, GAMMA, large_margin=False)
    # print(f"Loss: {loss}")

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()
    # print("Optimizer steped ahead")
    return loss