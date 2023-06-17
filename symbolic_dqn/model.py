import random
from collections import namedtuple, deque
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from expression_tree import *
from actions import add_feature_nodes, node_indices
import copy
import pickle #pickle for cloning environment

# Setting up a device
# print(f"Is GPU available: {torch.cuda.is_available()}")
# device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"

class Environment:
	'''
	The environment class is used to create an env object for the first markov decision process that is used to generate trees using DQN.
	This is a simpler version of the OpenAI gym environment. It returns next states via deterministic transitions and does not return a 
	reward upon state transition. The reward is instead relayed to the agent from a second MDP chained up to this one. The second environment
	is the actual gym env. 
	'''
	def __init__(self, main_env, node_vectors, node_instances, node_vector_dim, tree_depth=10):
		# state is a Multitree object 
		self.state = None 
		# main_env is an instance of the OpenAI gym environment (in this case, lunar lander)
		self.main_env = main_env
			
		# tree_full is a list containing the status of each tree in the multitree. If the tree is full (i.e no further nodes can be added) 
		self.tree_full = [False for _ in range(main_env.action_space.n)]
		self.done = False 
		self.tree_depth = tree_depth
		self.node_vector_dim = node_vector_dim
		self.main_env_steps_per_first_env_step = 20

		# Reset the main environment
		# self.main_env_state = main_env.reset()[0]
		# print(f"init shape: {self.main_env_state.shape}")

	def reset(self):
		n_trees = self.main_env.action_space.n # Having one tree per main env action
		self.done = False
		self.tree_full = [False for _ in range(self.main_env.action_space.n)]
		self.main_env_state = torch.from_numpy(self.main_env.reset()[0].reshape((1,-1))).float() #create tensor from numpy array for evaluation
		self.state = ExpressionMultiTree(self.tree_depth, n_trees, self.node_vector_dim)
		# print(f"reset POT {self.state.multitree_preorder_travs}")
		return self.state.vectorise_preorder_trav()

	def step(self, actions):
		# actions is a list of tree additions specified by the individual policies corresponding to each tree in the multitree
		if not self.done:
			if False in self.tree_full:
				tree_full_before_update = self.tree_full
				self.tree_full = self.state.update(actions, node_instances)
				# print(self.state.multitree_preorder_travs)
				# print(self.state.multitree.children[0]._children)
				# print(self.state.multitree.children[0]._children[1]._children) 

				rewards = np.array([0,0,0,0])

				# Stepping through k steps of the main env to evaluate the current multi-tree
				# count = 0
				for _ in range(self.main_env_steps_per_first_env_step):
					if not self.done:
						state_eval = self.state.evaluate(self.main_env_state)
						main_env_action = select_main_env_action(state_eval)
						# observation, reward, self.done, _, _ = self.main_env.step(main_env_action)
						observation, reward, done, _, _ = self.main_env.step(main_env_action)
						if done:
							# If the main env is completed, reset it
							observation = self.main_env.reset()[0]
						self.main_env_state = torch.from_numpy(observation.reshape((1,-1))).float()
						rewards[main_env_action] += reward
						# count += 1

				# print(count)
				# rewards += count

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

	#def append(self, *args):
	def push(self, *args):
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
	def forward(self, x, for_optimization=True, batch_size=None):
		# When using for optimization, a batch of inputs is passed in. In this case, reshape. When using for selecting actions, only one state is 
		# passed. In this case, the shape is already correctly set. Hence no reshaping is needed.

		# This option enables the model to accept an arbitrary batch size. 
		# This is useful when sampling a batch of transitions smaller than the preset batch_size (when replay memory has less than batch_size transitions)
		if batch_size == None:
			batch_size = self.batch_size

		if not for_optimization:
			non_op_batch_size = 1
			x = torch.reshape(x, (non_op_batch_size, -1))
		else:
			x = torch.reshape(x, (batch_size, -1))

		x = F.relu(self.layer1(x))
		x = F.relu(self.layer2(x))
		return self.layer3(x)



# Defining the temporal difference loss function
class DQN_Loss(nn.Module):
	def __init__(self):
		super(DQN_Loss, self).__init__()


	def forward(self, policy_net, target_net, states, actions, rewards, next_states, GAMMA, BATCH_SIZE):
		# 1-step TD loss
		with torch.no_grad():
			next_state_max = torch.max(target_net(next_states, batch_size = BATCH_SIZE), dim=1).values

		targets = rewards + GAMMA * next_state_max
		values = policy_net(states, batch_size = BATCH_SIZE).gather(1,actions.view(-1,1)).view(-1,)
		return F.mse_loss(values, targets)


# Defining epsilon greedy action selection
def select_action(states, EPS, policy_nets, node_instances):
	actions = []
	action_names = list(node_instances.keys())

	for i in range(len(states)):
		state = states[i]
		policy_net = policy_nets[i]
		state = state.to(device)
		sample = random.random()
		if sample > EPS:
			# print("Exploiting")
			with torch.no_grad():
				action_idx = torch.argmax(policy_net(state, for_optimization=False), dim=1).item()
				actions.append(action_names[action_idx]) #TODO: find out why list index is sometimes out of range

		else:
			# print("Exploring")
			actions.append(random.choice(action_names))

	return(actions)

# Defining softmax actions selection for the main environment
def select_main_env_action(state_eval):
	actions = [0,1,2,3]
	# probabilities = F.softmax((state_eval - state_eval.max), dim=0)
	# probabilities = probabilities.detach().numpy().flatten()
	# probabilities /= probabilities.sum()

	# return np.random.choice(actions, p=probabilities)

	soft = nn.Softmax(dim=-1)
	probabilities = soft((state_eval/state_eval.max()))
	probabilities = probabilities.cpu().detach().numpy()[0]
	if True in np.isnan(probabilities):
		probabilities = np.array([0.25, 0.25, 0.25, 0.25])
		
	return np.random.choice(actions, p=probabilities)



# Defining the optimization for the Q-network
def optimize_model(optimizers, policy_nets, target_nets, replay_memories, dqn_loss, node_indices, BATCH_SIZE = 32, GAMMA=0.99):
	'''
	Optimize the Q-networks using the agent's replay memory. 
	'''
	losses = []

	for i in range(len(replay_memories)):
		replay_memory = replay_memories[i]
		
		# Sample the whole replay memory if it has fewer than BATCH_SIZE number of transitions
		if len(replay_memory) < BATCH_SIZE:
			batch_size = len(replay_memory)
		else:
			batch_size = BATCH_SIZE

		policy_net = policy_nets[i]
		target_net = target_nets[i]
		optimizer = optimizers[i]

		# print("Sampling from agent's replay memory")
		batch_transitions = replay_memory.sample(batch_size)

		batch_states = []
		batch_actions = []
		batch_rewards = []
		batch_next_states = []
		batch_dones = []

		for i in range(batch_size):
			batch_states.append(batch_transitions[i].state)
			batch_next_states.append(batch_transitions[i].next_state)
			batch_rewards.append(batch_transitions[i].reward)
			batch_actions.append(node_indices[batch_transitions[i].action])

		batch_states = torch.stack(batch_states)
		batch_next_states = torch.stack(batch_next_states)
		batch_actions = torch.tensor(batch_actions)
		batch_rewards = torch.tensor(np.array(batch_rewards), dtype=torch.float32, requires_grad=True)
		batch_dones = torch.tensor(np.array(batch_dones))


		batch_states, batch_next_states, batch_rewards, batch_actions = batch_states.to(device), batch_next_states.to(device), batch_rewards.to(device), batch_actions.to(device)
		loss = dqn_loss(policy_net, target_net, batch_states, batch_actions, batch_rewards, batch_next_states, GAMMA, batch_size)
		# print(f"Loss: {loss}")
		losses.append(loss)

		# Optimize the model
		optimizer.zero_grad()
		loss.backward()
		# In-place gradient clipping
		torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
		optimizer.step()
		# print("Optimizer stepped ahead")
	
	return losses