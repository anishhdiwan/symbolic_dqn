# This stores a dictionary with keys as node symbol names and values as their corresponding arity for vectorization 
# The nodes are vectorised into two dimensional vectors dim 1 being a random number (for now) and dim 2 being the arity of the node

import random

node_vectors = {
	
'+': [random.sample(), 2],
'-': [random.sample(), 2],
'*': [random.sample(), 2],
'/': [random.sample(), 2],
'**2': [random.sample(), 1],
'sqrt': [random.sample(), 1],
'log': [random.sample(), 1],
'exp': [random.sample(), 1],
'sin': [random.sample(), 1],
'cos': [random.sample(), 1],
'max': [random.sample(), 2],
'min': [random.sample(), 2],
'x_': [random.sample(), 0],
'const': [random.sample(), 0],

}

actions_names = list(node_vectors.keys())