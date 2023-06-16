node_instances = {
	
'+': node_impl.Plus(),
'-': node_impl.Minus(),
'*': node_impl.Times(),
'/': node_impl.Div(),
'**2': node_impl.Square(),
'sqrt': node_impl.Sqrt(),
'log': node_impl.Log(),
# 'exp': node_impl.Exp(),
# 'sin': node_impl.Sin(),
# 'cos': node_impl.Cos(),
# 'max': node_impl.Max(),
# 'min': node_impl.Min(),
# 'x_': node_impl.Feature(),
'const?': node_impl.Constant()

}

def update_tree(tree_root_node, action):
	'''
	update_tree adds a new node (action) to a currently existing tree as per the pre-order traversal order
	'''
	# a boolean var to check if a child was added. If after running this function, no child was added then the tree is saturated and the episode needs to end
	child_added = False 

	if (tree_root_node.arity == 0):
		return child_added

	if (tree_root_node.arity > 0) and (len(tree_root_node._children) == 0):
		# if the current node has an arity > 0 and has no children then insert a child node at index 0 (left side)
		tree_root_node.insert_child(action, 0)
		child_added = True

	elif tree_root_node.arity - len(tree_root_node._children) == 1: 
		# if the current node can accomodate one more child 
		# this can happen for arity 1 nodes with no children or for arity 2 nodes with one child

		if len(tree_root_node._children) == 0:
			# if the node has no children then add a child node at index 0 (left side)
			tree_root_node.insert_child(action, 0)
			child_added = True
		else:
			# if the node has a child (which can only be a left child) then repeat for that child. 
			# if something was still not added (which can happen if the whole left branch is full) then add a right child
			child_added = update_tree(tree_root_node._children[0], action)
			if not child_added:
				tree_root_node.insert_child(action, 1)

	elif tree_root_node.arity - len(tree_root_node._children) == 0:
		# if the current node already has its max possible children then repeat for both children
		for child in tree_root_node._children:
			child_added = update_tree(child, action)

	return child_added