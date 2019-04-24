import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import setup as stp

N_CLASSES = stp.num_classes
N_OUT_CLASSES = stp.num_out_classes

def debugger(correct_pred, logits, y_exp, x, second_choice = False):
	print (str(int(correct_pred[0])) + " out of " + str(N_CLASSES))
	# To check first and second choice
	if (second_choice):
		out = list()
		for j in range(N_CLASSES):
			f_max = np.argmin(logits[0][j])
			f_max_2 = np.argmin(logits[0][j])
			for k in range(N_CLASSES):
				if logits[j][0][k] > logits[j][0][f_max]:
					f_max_2 = f_max
					f_max = k
				elif logits[j][0][k] > logits[j][0][f_max_2]:
					f_max_2 = k
			out.append((f_max, f_max_2))
		print out

def print_1by1(arr, title, n_classes):
	line = ""
	for i in range(n_classes):
		line += (str(arr[i]) + " ")
	print (title + line)

# Specific to sorting
def print_barchart(arr, expect, actual, figname):
	n_groups = len(arr[:N_CLASSES])
	fig, ax = plt.subplots()
	index = np.arange(n_groups)
	xticks = map(str, np.arange(1, n_groups + 1))
	bar_width = 0.35
	opacity = 0.8

	# expect_t, actual_t, expect_f, actual_f
	expect_t = []
	e_index_t = []
	expect_f = []
	e_index_f = []

	actual_t = []
	a_index_t = []
	actual_f = []
	a_index_f = []

	start_index = 0
	for e, a in zip(expect, actual):
		if e == a:
			expect_t.append(e+1)
			actual_t.append(a+1)
			e_index_t.append(start_index)
			a_index_t.append(start_index)
		else:
			expect_f.append(e+1)
			actual_f.append(a+1)
			e_index_f.append(start_index)
			a_index_f.append(start_index)
		start_index += 1

	rects1 = plt.bar(np.array(e_index_t), expect_t, bar_width, alpha=opacity, color='b', label='Expected Correct')
	rects2 = plt.bar(np.array(e_index_f), expect_f, bar_width, alpha=opacity, color='y', label='Expected Wrong')
	rects3 = plt.bar(np.array(a_index_t) + bar_width, actual_t, bar_width, alpha=opacity, color='g', label='Actual Correct')
	rects4 = plt.bar(np.array(a_index_f) + bar_width, actual_f, bar_width, alpha=opacity, color='r', label='Actual Wrong')
	
	blue_patch = mpatches.Patch(color='b', label='Expected Correct')
	yellow_patch = mpatches.Patch(color='y', label='Expected Wrong')
	green_patch = mpatches.Patch(color='g', label='Actual Correct')
	red_patch = mpatches.Patch(color='r', label='Actual Wrong')

	plt.xlabel('Element')
	plt.ylabel('Value')
	plt.title('Expected vs. actual labels')
	plt.xticks(index + bar_width, xticks)
	plt.legend(bbox_to_anchor=(1.14, 1.14), handles = [blue_patch, yellow_patch, green_patch, red_patch])
	plt.savefig('./data/' + figname)
	plt.close()

def check_perm_validity(arr, expect, actual):
	all_nums = []
	all_counts = []
	for i in range(N_OUT_CLASSES):
		all_nums.append(i)
		all_counts.append(actual.count(i))
	print all_nums
	print all_counts
	print "======"
	for e, a in zip(expect, actual):
		print e, a

# ASM plot (accuracy vs. scalability vs. models)
def print_acc_scale_models():
	ns = [6, 8, 9, 10, 11, 12, 16, 20, 24]
	nn = [1.00, 1.00, 1.00, 0.95, 0.66, 0.44, 0.23, 0.07, 0.04]
	dt = [1.00, 0.99, 0.92, 0.75, 0.68, 0.60, 0.34, 0.23, 0.15]
	rd = [0.17, 0.12, 0.11, 0.10, 0.09, 0.08, 0.06, 0.05, 0.04]
	plt.title('Predict Sorted Order', fontsize = 18)
	plt.xlabel('# Elements', fontsize = 16)
	plt.ylabel('% Correctly Guessed', fontsize = 16)
	plt.plot(ns, nn, 'b', linewidth = 2.8, label = 'Neural Net')
	plt.plot(ns, dt, 'g', linewidth = 2.8, label = 'Decision Trees')
	plt.plot(ns, rd, 'y', linewidth = 2.8, label = 'Random')
	plt.legend()
	plt.savefig('./results/' + 'asm.png')
	plt.clf()

# ASD plot (accuracy vs. scalability vs. data type)
def print_acc_scale_data():
	ns = [6, 8, 9, 10, 11, 12, 16, 20]
	set_1 = [1.00, 1.00, 0.97, 0.88, 0.57, 0.41, 0.21, 0.05]
	set_2 = [1.00, 1.00, 1.00, 0.95, 0.66, 0.44, 0.23, 0.07]
	set_3 = [1.00, 0.99, 0.93, 0.83, 0.74, 0.61, 0.32, 0.16]
	plt.title('Order Relations', fontsize = 18)
	plt.xlabel('# Elements', fontsize = 16)
	plt.ylabel('% Correctly Guessed', fontsize = 16)
	plt.plot(ns, set_1, 'orange', linewidth = 2.8, label = 'Data')
	plt.plot(ns, set_2, 'b', linewidth = 2.8, label = 'Data and Order Relations')
	plt.plot(ns, set_3, 'r', linewidth = 2.8, label = 'Order Relations')
	plt.legend()
	plt.savefig('./results/' + 'asd.png')
	plt.clf()

# AD plot (accuracy vs. design)
def print_acc_design():
	ns = [10, 12, 16, 20, 24, 28, 30]
	set_1 = [1.00, 1.00, 0.77, 0.57, 0.40, 0.30, 0.28]
	set_2 = [1.00, 1.00, 1.00, 0.99, 0.80, 0.49, 0.36]
	plt.title('Design', fontsize = 18)
	plt.xlabel('# Elements', fontsize = 16)
	plt.ylabel('% Correctly Guessed', fontsize = 16)
	plt.plot(ns, set_1, 'orange', linewidth = 2.8, label = 'Data')
	plt.plot(ns, set_2, 'r', linewidth = 2.8, label = 'Order Relations')
	plt.legend()
	plt.savefig('./results/' + 'ad.png')
	plt.clf()

def print_pretty(correct_pred, logits, y_exp, x, epoch):
	out = list()
	y_pred = list()
	for j in range(N_OUT_CLASSES):
		y_pred.append(np.argmax(logits[j][0]))
	print_1by1(x[0], 'input: ', N_CLASSES)
	print_1by1(y_exp[0], 'expect:', N_OUT_CLASSES)
	print_1by1(y_pred, 'pred:  ', N_OUT_CLASSES)
	print_barchart(x[0], list(y_exp[0]), y_pred, ('labels_' + str(epoch) + '.png'))
	check_perm_validity(x[0], list(y_exp[0]), y_pred)


#print_barchart(list([10, 30, 20, 40, 50]), list([1, 3, 2, 4, 5]), list([1, 2, 3, 4, 5]), 'labels_0.png')
#print_acc_scale_models()
#print_acc_scale_data()
#print_acc_design()
