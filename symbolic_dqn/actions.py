# This stores a dictionary with keys as node symbol names and values as their corresponding arity for vectorization 
import sys
sys.path.append('../')

import random
from genepro import node_impl


# node_vectors = {
	
# '+': [random.random(), 2],
# '-': [random.random(), 2],
# '*': [random.random(), 2],
# '/': [random.random(), 2],
# '**2': [random.random(), 1],
# 'sqrt': [random.random(), 1],
# 'log': [random.random(), 1],
# # 'exp': [random.random(), 1],
# # 'sin': [random.random(), 1],
# # 'cos': [random.random(), 1],
# # 'max': [random.random(), 2],
# # 'min': [random.random(), 2],
# #'x_': [random.random(), 0],
# 'const?': [random.random(), 0]

# }

# One hot vectors for nodes. Make sure that dimensionality of a vector = num_nodes + num features in env
node_vectors = {
	
'+': [1,0,0,0,0,0,0,0,0,0,0,0,0,0,2], 
'-': [0,1,0,0,0,0,0,0,0,0,0,0,0,0,2],
'*': [0,0,1,0,0,0,0,0,0,0,0,0,0,0,2],
'/': [0,0,0,1,0,0,0,0,0,0,0,0,0,0,2],
'**2': [0,0,0,0,1,0,0,0,0,0,0,0,0,0,1],
'sqrt': [0,0,0,0,0,1,0,0,0,0,0,0,0,0,1],
# 'log': [0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,2],
# 'exp': [0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,2],
# 'sin': [random.random(), 1],
# 'cos': [random.random(), 1],
# 'max': [random.random(), 2],
# 'min': [random.random(), 2],
#'x_': [random.random(), 0],
'const?': [0,0,0,0,0,0,1,0,0,0,0,0,0,0,0]

}


node_instances = {
	
'+': node_impl.Plus(),
'-': node_impl.Minus(),
'*': node_impl.Times(),
'/': node_impl.Div(),
'**2': node_impl.Square(),
'sqrt': node_impl.Sqrt(),
# 'log': node_impl.Log(),
# 'exp': node_impl.Exp(),
# 'sin': node_impl.Sin(),
# 'cos': node_impl.Cos(),
# 'max': node_impl.Max(),
# 'min': node_impl.Min(),
# 'x_': node_impl.Feature(),
'const?': node_impl.Constant()

}

# Make sure that these follow continuously from 0 -> num_nodes
node_indices = {

	'+': 0,
	'-': 1,
	'*': 2,
	'/': 3,
	'**2': 4,
	'sqrt': 5,
	# 'log': 5,
	# 'exp': 6,
	# 'sin': 8,
	# 'cos': 9,
	# 'max': 10,
	# 'min': 11,
	'const?':6
	# 'const_neg': 20,
	#'const_zero': 21,
	#'const_pos': 22

}


node_vector_dim = len(node_vectors['+'])
# template = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
def add_feature_nodes(node_vectors, node_instances, node_indices, main_env, template = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]):
	num_nodes = len(node_vectors)
	for i in range(main_env.observation_space.shape[0]):
		node_instances['x_' + str(i)] = node_impl.Feature(i)
		# node_vectors['x_' + str(i)] = [random.random(), 0]
		temp = template
		temp[num_nodes + i] = 1
		node_vectors['x_' + str(i)] = temp
		node_indices['x_' + str(i)] = i + num_nodes

	return node_vectors, node_instances, node_indices
