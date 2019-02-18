from skimage import io, img_as_float
from skimage.color import rgba2rgb
import numpy as np
import os
#from pptree import *
from ctree import *

def intersection(mask1, mask2):
	intersection = np.zeros(mask1.shape)
	intersection[:,:] = np.where(np.logical_and(mask1 != 0, mask2 != 0), np.ones(mask1.shape), np.zeros(mask1.shape))
	intersection_count = np.sum(intersection)
	return intersection_count

def get_top_left(mask):
	pix = np.where(mask == 1)
	rows = pix[0]
	cols = pix[1]
	return (np.min(rows), np.min(cols))

def get_top_left_bottom_right(mask):
	pix = np.where(mask == 1)
	rows = pix[0]
	cols = pix[1]
	return (np.min(rows), np.min(cols), np.max(rows), np.max(cols))

def get_pos_attr(mask):
	pix = np.where(mask == 1)
	rows = pix[0]
	cols = pix[1]
	return (np.min(rows) + np.min(cols))

def get_tree(masks, th = 0.7):

	for mask in masks:
		get_top_left(mask[1])

	# Create a directed acyclic graph
	edges = np.zeros((len(masks)+1, len(masks)+1))
	for i in range(len(masks)):
		for j in range(len(masks)):
			if i == j:
				continue
			mask_i = masks[i][1]
			mask_j = masks[j][1]
			mask_i_count = np.sum(mask_i)
			mask_j_count = np.sum(mask_j)
			min_count = min(mask_i_count, mask_j_count)
			max_count = max(mask_i_count, mask_j_count)
			intersection_count = intersection(mask_i, mask_j)
			#print (i, j)
			#print ('Pixel counts', intersection_count, min_count, intersection_count/float(min_count))
			if intersection_count/float(min_count) > th:
				a = i
				b = j
				if mask_i_count < mask_j_count:
					a = j
					b = i
				edges[a, b] = 1

	# Eliminate useless edges to get a tree
	for i in range(len(masks)):
		for j in range(len(masks)):
			for k in range(len(masks)):
				if edges[i,j] == 1 and edges[i,k] == 1 and edges[j,k] == 1:
					edges[i,k] = 0

	# All nodes which don't have a parent are attached to the body element
	# Create the root = body = background element
	masks.append(('body', np.ones(masks[0][1].shape)))
	for i in range(len(masks) - 1):
		has_parent = False
		for j in range(len(masks)):
			if edges[j,i] == 1:
				has_parent = True
				break
		if not has_parent:
			edges[len(masks) - 1, i] = 1

	# Initialize tree
	has_node = {} # whether node id has a node assigned to it
	for i in range(len(masks)):
		has_node[i] = (False, None)

	# Breadth first search
	total = 0
	bf = [len(masks) - 1]
	# + str(get_top_left_bottom_right(masks[-1][1]))
	root = Node(masks[-1][0] , get_pos_attr(masks[-1][1])) # class name, attr and parent
	total += 1
	has_node[len(masks) - 1] = (True, root) 
	start = 0
	while start < len(bf):
		cur_node = bf[start]
		cur_Node = has_node[cur_node][1]
		for i in range(len(masks)):
			new_node = i
			if edges[cur_node,new_node] == 1 and not has_node[new_node][0]:
				# + str(get_top_left_bottom_right(masks[new_node][1]))
				new_Node = Node(masks[new_node][0] , get_pos_attr(masks[new_node][1]), cur_Node) # class name, attr and parent
				total += 1
				has_node[new_node] = (True, new_Node) 
				bf.append(new_node)
		start += 1

	#print_tree(root)
	return root

# Views the tree for a give location
def view_tree(location):
	# Read image
	image = io.imread(location + 'images/' + img_id + '/' + img_id + '.png')

	# Read masks
	files = [file_name for file_name in os.listdir(location + 'masks/' + img_id + '/') if file_name.endswith('.png')]
	files = sorted(files, key = lambda x: x.split('_')[1])
	masks = [(file_name.split('_')[1].split('.')[0], img_as_float(io.imread(location + 'masks/' + img_id + '/' + file_name)))
		for file_name in files]

	# Print the tree for obtained masks
	tree = get_tree(masks)
	return tree

def compare(root1, root2):
	if root1.name != root2.name:
		print ('Different Classes', root1.name, root2.name)
		return -1

	children1 = sorted(root1.children, key = lambda node: node.attr)
	children2 = sorted(root2.children, key = lambda node: node.attr)
	if len(children1) != len(children2):
		print ('Different Number of Nodes')
		return 0

	for child1, child2 in zip(children1, children2):
		status = compare(child1, child2)
		if status != 1:
			return status
	return 1


# Which ID to view masked
img_id = str(3602)

# Ground-truth location
gt_location = './resized_data/'
val_location = './validation_data/'

total = 0
correct = 0
dif_num_el = 0
dif_class = 0
ids = [int(directory) for directory in os.listdir(val_location + 'images/')]
for i in range(len(ids)):
	img_id = str(ids[i])
	print (img_id)

	gt_tree = view_tree(gt_location)
	val_tree = view_tree(val_location)

	total += 1
	same = compare(gt_tree, val_tree)
	if same != 1:
		print ('Ground-truth')
		print_tree(gt_tree)
		print ('Predicted')
		print_tree(val_tree)

	if same == 1:
		correct += 1
	elif same == 0:
		dif_num_el += 1
	else:
		dif_class += 1

print ('Accuracy', correct/float(total))
print ('Diff Num Elements', dif_num_el/float(total))
print ('Dif Classes', dif_class/float(total))

