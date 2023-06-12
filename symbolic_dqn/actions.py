# This stores a dictionary with keys as node symbol names and values as their corresponding arity for vectorization 
# The nodes are vectorised into two dimensional vectors dim 1 being a random number (for now) and dim 2 being the arity of the node

import random


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
'const': [random.random(), 0],

}

actions_names = list(node_vectors.keys())