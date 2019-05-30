import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import setup as stp
import pickle

N_CLASSES = stp.num_classes
N_OUT_CLASSES = stp.num_out_classes

def debugger(correct_pred, logits, y_exp, x, second_choice = False):
	print (str(int(correct_pred[0])) + " out of " + str(N_OUT_CLASSES))
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

def debugger_whole_batch(correct_pred, logits, y_exp, x, step):
	lst = list()
	for i in range(len(correct_pred)):
		y_pred = list()
		for j in range(N_OUT_CLASSES):
			y_pred.append(np.argmax(logits[j][i]))
		# print_1by1(x[i], 'input: ', N_CLASSES)
		# print_1by1(y_exp[i], 'expect:', N_OUT_CLASSES)
		# print_1by1(y_pred, 'pred:  ', N_OUT_CLASSES)
		c_lst = list()
		c_lst.append(int(x[i][0]))
		c_lst.append(int(x[i][1]))
		c_lst.append(int(y_pred[0]))
		lst.append(c_lst)
	mark_dots(lst, step)

def mark_dots(arr, step):
	white_x, white_y = list(), list()
	black_x, black_y = list(), list()
	for e in arr:
		if e[2] == 0:
			white_x.append(e[0])
			white_y.append(e[1])
		else:
			black_x.append(e[0])
			black_y.append(e[1])
	plt.plot(white_x, white_y, 'ro')
	plt.plot(black_x, black_y, 'bo')
	plt.savefig('./results/comparator/sep_' + str(step) + '.png')
	if step % 1000 == 0:
		plt.close()

def print_1by1(arr, title, n_classes):
	line = ""
	for i in range(n_classes):
		line += (str(int(arr[i])) + " ")
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
	plt.savefig('./data/labels/' + figname)
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

def print_ltv(A, B, C, x, filename):
	plt.title('Loss and Accuracy', fontsize = 18)
	plt.xlabel('# Steps', fontsize = 16)
	plt.ylabel('% Value', fontsize = 16)
	plt.plot(x, A, 'blue', linewidth = 1.0, label = 'Loss')
	plt.plot(x, B, 'orange', linewidth = 1.0, label = 'Training Accuracy')
	plt.plot(x, C, 'red', linewidth = 1.0, label = 'Validation Accuracy')
	# Remove legend altogether
	#plt.legend(loc = 'upper left')
	plt.savefig('./results/loss_and_acc/' + filename)
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

def combine_plots(model_names, colors, target_metric, label_names, fig_name, title_name):
	dir_name = './data/stats/'
	plt.title(title_name, fontsize = 18)
	plt.xlabel('# Steps', fontsize = 16)
	for (model_name, color, label_name) in zip(model_names, colors, label_names):
		model_root = model_name + '_ml_'
		for metric in ['steps', 'losses', 't_accs', 'v_accs']:
			if metric == target_metric:
				val_filename = dir_name + model_root + metric + '.p'
				step_filename = dir_name + model_root + 'steps' + '.p'
				print 'Retrieve values from ' + val_filename
				seq = pickle.load(open(val_filename, 'r'))
				print seq
				steps = np.linspace(1, 100000, 100)
				plt.plot(steps, seq, color, linewidth = 1.8, label = label_name)
	plt.legend(loc = 'lower right')
	plt.ylim([0, 105])
	plt.savefig('./results/' + fig_name + '.png')
	plt.clf()

combine_plots(['a_10', 'ac_10', 'b_10', 'bc_10'], ['r', 'b', 'g', 'm'], 'v_accs', \
	['C-Baseline', 'C-Baseline (order rel)', 'C-Design', 'C-Design (order rel)'], 'asbs_10', 'Accuracy N = 10')

#print_barchart(list([10, 30, 20, 40, 50]), list([1, 3, 2, 4, 5]), list([1, 2, 3, 4, 5]), 'labels_0.png')
#print_acc_scale_models()
#print_acc_scale_data()
#print_acc_design()
