'''
Method to visualise a tree. This is independent of any multitree code and is purely for visualising trees
'''

# Defining a tree structure as per one of the trees generated by inference.py
pre_order = ['+', '0.0', '+', 'sqrt', '/', 'x_5', 'x_0', None, None, 'x_0', None]
iterator = iter(pre_order)
arities = {

'+':2,
'0.0':0,
'sqrt':1,
'/':2,
None:0,
'x_5':0,
'x_0':0
 }

# Manually adding nodes 

from PrettyPrint import PrettyPrintTree
from colorama import Back
from time import sleep
import os


class Node:
    def __init__(self, value, arity=0):
        self.val = value
        self.children = []
        self.arity = arity

    def add_child(self, child):
        self.children.append(child)
        return child


os.system('cls')

pt = PrettyPrintTree(lambda x: x.children, lambda x: x.val, color=Back.BLACK)
root = Node('+')
pt(root)
print(''' 
	''')
sleep(1)
os.system('cls')

c1 = root.add_child(Node('0.0'))
pt(root)
print(''' 
	''')
sleep(1)
os.system('cls')

c2 = root.add_child(Node('+'))
pt(root)
print(''' 
	''')
sleep(1)
os.system('cls')

c3 = c2.add_child(Node('sqrt'))
pt(root)
print(''' 
	''')
sleep(1)
os.system('cls')

c4 = c3.add_child(Node('/'))
pt(root)
print(''' 
	''')
sleep(1)
os.system('cls')

c5 = c4.add_child(Node('x_0'))
pt(root)
print(''' 
	''')
sleep(1)
os.system('cls')

c6 = c4.add_child(Node('x_5'))
pt(root)
print(''' 
	''')
sleep(1)
os.system('cls')

c7 = c2.add_child(Node('max'))
pt(root)
print(''' 
	''')
sleep(1)
os.system('cls')

c8 = c7.add_child(Node('**2'))
pt(root)
print(''' 
	''')
sleep(1)
os.system('cls')

c10 = c8.add_child(Node('const'))
pt(root)
print(''' 
	''')
sleep(1)
os.system('cls')

c9 = c7.add_child(Node('x_3'))
pt(root)
print(''' 
	''')
sleep(1)





# class Node:
#     def __init__(self, value, arity=0):
#         self.val = value
#         self.children = []
#         self.arity = arity

#     def add_child(self, child, index):
#         self.children.insert(index, child)
#         return child



# def update_tree(tree_root_node, action):
# 	'''
# 	update_tree adds a new node (action) to a currently existing tree as per the pre-order traversal order
# 	'''
# 	# a boolean var to check if a child was added. If after running this function, no child was added then the tree is saturated and the episode needs to end
# 	child_added = False 

# 	if (tree_root_node.arity == 0):
# 		return child_added

# 	if (tree_root_node.arity > 0) and (len(tree_root_node.children) == 0):
# 		# if the current node has an arity > 0 and has no children then insert a child node at index 0 (left side)
# 		tree_root_node.add_child(Node(action, arities[action]), 0)
# 		child_added = True

# 	elif tree_root_node.arity - len(tree_root_node.children) == 1: 
# 		# if the current node can accomodate one more child 
# 		# this can happen for arity 1 nodes with no children or for arity 2 nodes with one child

# 		if len(tree_root_node.children) == 0:
# 			# if the node has no children then add a child node at index 0 (left side)
# 			tree_root_node.add_child(Node(action, arities[action]), 0)
# 			child_added = True
# 		else:
# 			# if the node has a child (which can only be a left child) then repeat for that child. 
# 			# if something was still not added (which can happen if the whole left branch is full) then add a right child
# 			child_added = update_tree(tree_root_node.children[0], action)
# 			if not child_added:
# 				tree_root_node.add_child(Node(action, arities[action]), 1)

# 	elif tree_root_node.arity - len(tree_root_node.children) == 0:
# 		# if the current node already has its max possible children then repeat for both children
# 		for child in tree_root_node.children:
# 			print(child.val)
# 			child_added = update_tree(child, action)

# 	return child_added



# pt = PrettyPrintTree(lambda x: x.children, lambda x: x.val, color=Back.BLACK)

# root = Node('+', 2)
# root.add_child(Node('0.0'), 0)
# root.add_child(Node('+'), 2)



# child_added = True

# for action in iterator:
# 	# print(action)
# 	# while child_added:
# 	child_added = update_tree(root, Node(action, arities[action]))
# 	pt(root)
# 	print(''' 
# 		''')
# 	sleep(1)