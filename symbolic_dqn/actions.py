# This stores a dictionary with keys as node symbol names and values as their corresponding arity for vectorization 
# The nodes are vectorised into two dimensional vectors dim 1 being a random number (for now) and dim 2 being the arity of the node
import sys
sys.path.append('../')

import random
from genepro import node_impl



node_vector_dim = 2
node_vectors = {
	
'+': [random.random(), 2],
'-': [random.random(), 2],
'*': [random.random(), 2],
'/': [random.random(), 2],
'**2': [random.random(), 1],
'sqrt': [random.random(), 1],
'log': [random.random(), 1],
'exp': [random.random(), 1],
'sin': [random.random(), 1],
'cos': [random.random(), 1],
'max': [random.random(), 2],
'min': [random.random(), 2],
'x_': [random.random(), 0],
'const?': [random.random(), 0]

}


node_instances = {
	
'+': node_impl.Plus(),
'-': node_impl.Minus(),
'*': node_impl.Times(),
'/': node_impl.Div(),
'**2': node_impl.Square(),
'sqrt': node_impl.Sqrt(),
'log': node_impl.Log(),
'exp': node_impl.Exp(),
'sin': node_impl.Sin(),
'cos': node_impl.Cos(),
'max': node_impl.Max(),
'min': node_impl.Min(),
# 'x_': node_impl.Feature(),
'const?': node_impl.Constant()

}

def add_feature_nodes(node_vectors, node_instances, main_env):
	for i in range(main_env.observation_space.shape[0]):
		node_instances['x_' + str(i)] = node_impl.Feature(i)
		node_vectors['x_' + str(i)] = [random.random(), 0]

	return node_vectors, node_instances

