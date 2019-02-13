import tensorflow as tf
import numpy as np

N_CLASSES = 10

def debugger(correct_pred, logits, y_exp, x):
	print (str(int(correct_pred[0])) + " out of " + str(N_CLASSES))
	# to check first and second choice
	# out = list()
	# for j in range(N_CLASSES):
	# 	f_max = np.argmin(logits[0][j])
	# 	f_max_2 = np.argmin(logits[0][j])
	# 	for k in range(N_CLASSES):
	# 		if logits[j][0][k] > logits[j][0][f_max]:
	# 			f_max_2 = f_max
	# 			f_max = k
	# 		elif logits[j][0][k] > logits[j][0][f_max_2]:
	# 			f_max_2 = k
	# 	out.append((f_max, f_max_2))
	# print out

def print_1by1(arr, title):
	line = ""
	for i in range(N_CLASSES):
		line += (str(arr[i]) + " ")
	print (title + line)

def pretty_printing(correct_pred, logits, y_exp, x):
	out = list()
	y_pred = list()
	for j in range(N_CLASSES):
		y_pred.append(np.argmax(logits[j][0]))
	print_1by1(x[0], 'input: ')
	print_1by1(y_exp[0], 'expect:')
	print_1by1(y_pred, 'pred:  ')