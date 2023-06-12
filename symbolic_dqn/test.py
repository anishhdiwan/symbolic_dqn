import numpy as np
# import torch
# from actions import *


# tree_depth = 4
# multitree_preorder_travs = [

# ['+', '0'],
# ['+', '0', '-'],
# ['+', '0', 'log']

# ]

# def vectorise_preorder_trav():
#     # Turn the preorder traversal of the tree (list of nodes that are operator tokens) into a vector representation
#     vectorised_multitree_preorder_trav = []
#     for trav in multitree_preorder_travs:
#         vectorised_trav = np.zeros((2**tree_depth - 1, 2))
#         for i in range(len(trav)):
#             operator = trav[i]

#             if operator.replace(".", "").replace("-","").isnumeric():
#                 vectorised_trav[i] = np.array(node_vectors['const'])
#             elif operator[:2] == "x_":
#                 vectorised_trav[i] = np.array(node_vectors['x_'])
#             else:
#                 vectorised_trav[i] = np.array(node_vectors[operator])

#         vectorised_multitree_preorder_trav.append(torch.tensor(vectorised_trav, dtype=torch.float32, requires_grad=True))

#     return vectorised_multitree_preorder_trav


# temp = vectorise_preorder_trav()
# print(temp)
# print(temp[0])

print(len(np.array([1,2,3])))