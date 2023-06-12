import random
from collections import namedtuple, deque
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from expression_tree import *
from actions import *
import copy

device = "cuda"


class Environment:
	'''
	The environment class is used to create an env object for the first markov decision process that is used to generate trees using DQN.
	This is a simpler version of the OpenAI gym environment. It returns next states via deterministic transitions and does not return a 
	reward upon state transition. The reward is instead relayed to the agent from a second MDP chained up to this one. The second environment
	is the actual gym env. 
	'''
	def __init__(self, main_env, tree_depth=10):
		# state is a Multitree object 
		self.state = None 
		# main_env is an instance of the OpenAI gym environment (in this case, lunar lander)
		self.main_env = main_env
			
		# tree_full is a list containing the status of each tree in the multitree. If the tree is full (i.e no further nodes can be added) 
		self.tree_full = [False for _ in range(main_env.action_space.n)]
		self.done = False 
		self.tree_depth = tree_depth

		# Reset the main environment
		self.main_env_state = main_env.reset()[0]

	def reset(self):
		n_trees = self.main_env.action_space.n # Having one tree per main env action
		self.state = ExpressionMultiTree(self.tree_depth, n_trees)
		return self.state.vectorise_preorder_trav()

	def step(self, actions):
		# actions is a list of tree additions specified by the individual policies corresponding to each tree in the multitree
		if not self.done:
			if False in self.tree_full:
				tree_full_before_update = self.tree_full
				self.tree_full = self.state.update(actions)

				state_eval = self.state.evaluate(self.main_env_state)
				main_env_action = select_main_env_action(state_eval)

				rewards = []
				for action in range(self.main_env.action_space.n):
					copy_env = copy.deepcopy(self.main_env)
					_, reward, done, _ = copy_env.step(action)
					rewards.append(reward)

				self.main_env_state, _, _, self.done = self.main_env.step(main_env_action) 

				if not False in self.tree_full:
					self.done = True

				return self.state.vectorise_preorder_trav(), rewards, self.done, tree_full_before_update
			
			else:
				self.done = True
				print("All trees are full. No new additions possible")
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

	def __init__(self, capacity=500):
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
		super(DQN, self).__init__()
		self.layer1 = nn.Linear(n_observation_feats, 32)
		self.layer2 = nn.Linear(32, 64)
		self.layer3 = nn.Linear(64, n_actions)
		self.batch_size = batch_size

	# Called with either one element to determine next action, or a batch
	def forward(self, x, for_optimization=True):
		# When using for optimization, a batch of inputs is passed in. In this case, reshape. When using for selecting actions, only one state is 
		# passed. In this case, the shape is already correctly set. Hence no reshaping is needed.
		if not for_optimization:
			self.batch_size = 1
		x = torch.reshape(x, (self.batch_size,-1))
		x = F.relu(self.layer1(x))
		x = F.relu(self.layer2(x))
		return self.layer3(x)



# Defining the temporal difference loss function
class DQN_Loss(nn.Module):
	def __init__(self):
		super(DQN_Loss, self).__init__()


	def forward(self, policy_net, target_net, states, actions, rewards, next_states, dones, GAMMA):
		# 1-step TD loss
		with torch.no_grad():
			next_state_max = torch.max(target_net(next_states), dim=1).values

		targets = rewards + GAMMA * next_state_max 
		values = policy_net(states).gather(1,actions.view(-1,1)).view(-1,)

		return F.mse_loss(values, targets)


# Defining epsilon greedy action selection
def select_action(states, EPS, policy_nets):
	actions = []

	for i in range(len(states)):
		state = states[i]
		policy_net = policy_nets[i]
		state = state.to(device)
		sample = random.random()
		if sample > EPS:
			# print("Exploiting")
			with torch.no_grad():
				action_idx = torch.argmax(policy_net(state, for_optimization=False), dim=1).item()
				actions.append(action_names[action_idx])

		else:
			# print("Exploring")
			actions.append(random.choice(action_names))

	return(actions)

# Defining softmax actions selection for the main environment
def select_main_env_action(state_eval):
	actions = [0,1,2,3]
	probabilities = F.softmax(state_eval)
	return np.random.choice(actions, p=probabilities)



# Defining the optimization for the Q-network
def optimize_model(optimizers, policy_nets, target_nets, replay_memories, dqn_loss, BATCH_SIZE = 32, GAMMA=0.99):
	'''
	Optimize the Q-networks using the agent's replay memory. 
	'''
	losses = []

	for i in range(len(replay_memories)):
		replay_memory = replay_memories[i]
		policy_net = policy_nets[i]
		target_net = target_nets[i]
		optimizer = optimizers[i]

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
		loss = dqn_loss(policy_net, target_net, batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones, GAMMA)
		# print(f"Loss: {loss}")
		losses.append(loss)

		# Optimize the model
		optimizer.zero_grad()
		loss.backward()
		# In-place gradient clipping
		torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
		optimizer.step()
		# print("Optimizer steped ahead")
	
	return losses