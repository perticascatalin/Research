import numpy as np
import matplotlib.pyplot as plt
import setup as stp

N_CLASSES = stp.num_classes()

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

def print_barchart(arr, expect, actual, figname):
	n_groups = len(arr[:N_CLASSES])
	fig, ax = plt.subplots()
	index = np.arange(n_groups)
	xticks = map(str, np.arange(1, n_groups + 1))
	bar_width = 0.35
	opacity = 0.8

	rects1 = plt.bar(index, tuple(expect), bar_width, alpha=opacity, color='g', label='Expected')
	rects2 = plt.bar(index + bar_width, tuple(actual), bar_width, alpha=opacity, color='r', label='Actual')
	 
	plt.xlabel('Element')
	plt.ylabel('Value')
	plt.title('Expected vs. actual labels')
	plt.xticks(index + bar_width, tuple(xticks))
	plt.legend()
	plt.savefig('./results/' + figname)

def print_acc_scale_models():
	ns = [8, 10, 12, 16, 20]
	nn = [1.00, 0.95, 0.44, 0.23, 0.07]
	dt = [0.99, 0.75, 0.60, 0.34, 0.23]
	rd = [0.12, 0.10, 0.08, 0.06, 0.05]
	plt.plot(ns, nn, 'b', ns, dt, 'g', ns, rd, 'y')
	plt.show()

def pretty_printing(correct_pred, logits, y_exp, x, epoch):
	out = list()
	y_pred = list()
	for j in range(N_CLASSES):
		y_pred.append(np.argmax(logits[j][0]))
	print_1by1(x[0], 'input: ')
	print_1by1(y_exp[0], 'expect:')
	print_1by1(y_pred, 'pred:  ')
	print_barchart(x[0], list(y_exp[0]), y_pred, ('labels_' + str(epoch) + '.png'))

#print_barchart(list([10, 30, 20, 40, 50]), list([1, 3, 2, 4, 5]), list([1, 2, 3, 4, 5]), 'labels_0.png')
print_acc_scale_models()
