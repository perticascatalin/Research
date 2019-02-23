import numpy as np
import matplotlib.pyplot as plt

N_CLASSES = 20

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

def print_barchart(arr, expect, actual):
	# data to plot
	n_groups = len(arr)
	 
	# create plot
	fig, ax = plt.subplots()
	index = np.arange(n_groups)
	bar_width = 0.35
	opacity = 0.8
	 
	rects1 = plt.bar(index, expect, bar_width,
	alpha=opacity,
	color='b',
	label='Expected')
	 
	rects2 = plt.bar(index + bar_width, actual, bar_width,
	alpha=opacity,
	color='g',
	label='Actual')
	 
	plt.xlabel('Element')
	plt.ylabel('Value')
	plt.title('Scores by person')
	plt.xticks(index + bar_width, ('1', '2', '3', '4', '5'))
	plt.legend()
	
	plt.tight_layout()
	plt.show()

def pretty_printing(correct_pred, logits, y_exp, x):
	out = list()
	y_pred = list()
	for j in range(N_CLASSES):
		y_pred.append(np.argmax(logits[j][0]))
	print_1by1(x[0], 'input: ')
	print_1by1(y_exp[0], 'expect:')
	print_1by1(y_pred, 'pred:  ')
	print_barchart(x[0], y_exp[0], y_pred)

print_barchart(list([10, 30, 20, 40, 50]), list([1, 3, 2, 4, 5]), list([1, 2, 3, 4, 5]))

