import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D
import config as conf
import pickle

N_CLASSES = conf.num_inputs
N_OUT_CLASSES = conf.num_outputs

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

def debugger_whole_batch_cuboid(correct_pred, logits, y_exp, x, step):
	lst = list()
	for i in range(len(correct_pred)):
		y_pred = list()
		for j in range(N_OUT_CLASSES):
			y_pred.append(np.argmax(logits[i]))
		# print_1by1_asis(x[i], 'input: ', N_CLASSES)
		# print_1by1(y_exp[i], 'expect:', N_OUT_CLASSES)
		# print_1by1(y_pred, 'pred:  ', N_OUT_CLASSES)
	 	c_lst = list()
	 	c_lst.append(x[i][0])
	 	c_lst.append(x[i][1])
	 	c_lst.append(x[i][2])
		c_lst.append(int(y_pred[0])) # for predicted
		#c_lst.append(int(y_exp[i])) # for ground-truth
		lst.append(c_lst)
	mark_dots_cube(lst, step)

def mark_dots_cube(arr, step):
	red_x, red_y, red_z = list(), list(), list()
	blue_x, blue_y, blue_z = list(), list(), list()
	green_x, green_y, green_z = list(), list(), list()
	yellow_x, yellow_y, yellow_z = list(), list(), list()
	for e in arr:
		if e[3] == 0:
			red_x.append(e[0])
			red_y.append(e[1])
			red_z.append(e[2])
		elif e[3] == 1:
			blue_x.append(e[0])
			blue_y.append(e[1])
			blue_z.append(e[2])
		elif e[3] == 2:
			green_x.append(e[0])
			green_y.append(e[1])
			green_z.append(e[2])
		elif e[3] == 3:
			yellow_x.append(e[0])
			yellow_y.append(e[1])
			yellow_z.append(e[2])

	ax = plt.subplot(111, projection='3d')
	ax.scatter(red_x, red_y, red_z, c = 'red')
	ax.scatter(blue_x, blue_y, blue_z, c = 'blue')
	ax.scatter(green_x, green_y, green_z, c = 'green')
	ax.scatter(yellow_x, yellow_y, yellow_z, c = 'yellow')
	plt.savefig('./results/counter/sep_' + str(step) + '.png')
	if step % 1000 == 0:
		plt.close()


def print_1by1(arr, title, n_classes):
	line = ""
	for i in range(n_classes):
		line += (str(int(arr[i])) + " ")
	print (title + line)

def print_1by1_asis(arr, title, n_classes):
	line = ""
	for i in range(n_classes):
		line += (str(round(arr[i], 3)) + " ")
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

def print_ltv(A, B, C, D, x, filename):
	plt.title('Loss and Accuracy', fontsize = 18)
	plt.xlabel('# Steps', fontsize = 16)
	plt.ylabel('% Value', fontsize = 16)
	plt.plot(x, A, 'blue', linewidth = 1.0, label = 'Training Loss')
	plt.plot(x, B, 'cyan', linewidth = 1.0, label = 'Validation Loss')
	plt.plot(x, C, 'orange', linewidth = 1.0, label = 'Training Accuracy')
	plt.plot(x, D, 'red', linewidth = 1.0, label = 'Validation Accuracy')
	# Remove legend altogether
	#plt.legend(loc = 'upper left')
	plt.savefig('./results/loss_and_acc/' + filename)
	plt.clf()

def print_pretty(correct_pred, logits, y_exp, x, epoch, count_correct = False):
	out = list()
	y_pred = list()
	for j in range(N_OUT_CLASSES):
		y_pred.append(np.argmax(logits[j][0]))
	print_1by1(x[0], 'input: ', N_CLASSES)
	print_1by1(y_exp[0], 'expect:', N_OUT_CLASSES)
	print_1by1(y_pred, 'pred:  ', N_OUT_CLASSES)
	print_barchart(x[0], list(y_exp[0]), y_pred, ('labels_' + str(epoch) + '.png'))
	check_perm_validity(x[0], list(y_exp[0]), y_pred)

	if count_correct:
		num_correct = 0
		num_all = 100
		for i in range(num_all):
			print (str(int(correct_pred[i])) + " out of " + str(N_OUT_CLASSES))
			if correct_pred[i] == N_OUT_CLASSES:
				num_correct+=1
			out = list()
			y_pred = list()
			for j in range(N_OUT_CLASSES):
				y_pred.append(np.argmax(logits[j][i]))
			print_1by1(x[i], 'input: ', N_CLASSES)
			print_1by1(y_exp[i], 'expect:', N_OUT_CLASSES)
			print_1by1(y_pred, 'pred:  ', N_OUT_CLASSES)
			print_barchart(x[i], list(y_exp[i]), y_pred, ('labels_' + str(epoch) + '.png'))
			check_perm_validity(x[i], list(y_exp[i]), y_pred)
		print 'Number of correctly sorted:', num_correct

def combine_plots(model_names, label_names, colors, target_metric, fig_name, title_name, loc):
	dir_name = './data/stats/'
	plt.title(title_name, fontsize = 18)
	plt.xlabel('# Steps', fontsize = 16)
	for (model_name, color, label_name) in zip(model_names, colors, label_names):
		model_root = model_name + '_ml_'
		for metric in ['steps', 't_losses', 'v_losses', 't_accs', 'v_accs']:
			if metric == target_metric:
				val_filename = dir_name + model_root + metric + '.p'
				step_filename = dir_name + model_root + 'steps' + '.p'
				print 'Retrieve values from ' + val_filename
				seq = pickle.load(open(val_filename, 'r'))
				print seq
				# steps = np.linspace(1, 100000, 100)
				steps = np.linspace(1, 100000, 20)
				plt.plot(steps, seq, color, linewidth = 1.8, label = label_name)
	# plt.legend(loc = 'lower right')
	plt.legend(loc = loc)
	plt.ylim([0, 105])
	plt.savefig('./results/' + fig_name + '.png')
	plt.clf()

def print_pickle(filename):
	seq = pickle.load(open(filename, 'r'))
	print seq

# combine_plots(['a_10', 'ac_10', 'b_10', 'bc_10'], ['C-Baseline', 'C-Baseline (order rel)', 'C-Design', 'C-Design (order rel)'], \
#	['r', 'b', 'g', 'm'], 'v_accs', 'asbs_10', 'Accuracy N = 10')

# combine_plots(['b_10', 'bc_10', 'c_10', 'b_20', 'bc_20', 'c_20'], ['C-Design 10', 'C-Design (order rel) 10', 'C-FRN 10', 'C-Design 20', 'C-Design (order rel) 20', 'C-FRN 20'], \
#	['lightgreen', 'magenta', 'orange', 'green', 'blue', 'red'], 'v_accs', 'bscs_10_20', 'Accuracy N = 10, 20')

# combine_plots(['b_30', 'bc_30', 'c_30'], ['C-Design 30', 'C-Design (order rel) 30', 'C-FRN 30'], \
#	['lightgreen', 'green', 'blue'], 'v_accs', 'bscs_30', 'Accuracy N = 30')

#print_pickle('./data/stats/c_30_ml_v_accs.p')
#print_barchart(list([10, 30, 20, 40, 50]), list([1, 3, 2, 4, 5]), list([1, 2, 3, 4, 5]), 'labels_0.png')
#print_acc_scale_models()
#print_acc_scale_data()
#print_acc_design()

def combine_plots_n(N):
	model_names = ['base_data', 'base_or', 'Q', 'R']
	model_names = model_names if N == 30 else map(lambda x: x + '_' + str(N), model_names)
	displ_names = ['Baseline', 'Order Rel', 'Rel Net', 'Conv Rel Net']
	colors = ['r', 'g', 'b', 'm']

	fig_name = 'all_' + str(N) + '_acc'
	title_name = 'Accuracy N = ' + str(N)
	loc = 'upper left'
	combine_plots(model_names, displ_names, colors, 'v_accs', fig_name, title_name, loc)

	fig_name = 'all_' + str(N) + '_loss'
	title_name = 'Loss N = ' + str(N)
	loc = 'upper right'
	combine_plots(model_names, displ_names, colors, 'v_losses', fig_name, title_name, loc)

# combine_plots_n(30)

def combine_plots_n_temp(N):
	model_names = ['base_data', 'base_or', 'test']
	model_names = model_names if N == 30 else map(lambda x: x + '_' + str(N), model_names)
	displ_names = ['Baseline', 'Order Rel', 'Design']
	colors = ['r', 'g', 'c']

	fig_name = 'all_' + str(N) + '_acc'
	title_name = 'Accuracy N = ' + str(N)
	loc = 'upper left'
	combine_plots(model_names, displ_names, colors, 'v_accs', fig_name, title_name, loc)

	fig_name = 'all_' + str(N) + '_loss'
	title_name = 'Loss N = ' + str(N)
	loc = 'upper right'
	combine_plots(model_names, displ_names, colors, 'v_losses', fig_name, title_name, loc)

# combine_plots_n_temp(25)

# Bla bla plot
def print_acc_all():
	ns = [10, 15, 20, 25, 30]
	set_1 = [1.00, 1.00, 0.77, 0.57, 0.40]
	set_2 = [1.00, 1.00, 1.00, 0.99, 0.80]
	set_3 = [1.00, 1.00, 1.00, 0.99, 0.80]
	set_4 = [1.00, 1.00, 1.00, 0.99, 0.80]
	plt.title('Accuracy by Model', fontsize = 18)
	plt.xlabel('# Elements', fontsize = 16)
	plt.ylabel('% Correctly Guessed', fontsize = 16)
	plt.plot(ns, set_1, 'red', linewidth = 2.8, label = 'Baseline')
	plt.plot(ns, set_2, 'green', linewidth = 2.8, label = 'Order Rel')
	plt.plot(ns, set_3, 'blue', linewidth = 2.8, label = 'Rel Net')
	plt.plot(ns, set_4, 'magenta', linewidth = 2.8, label = 'Conv Rel Net')
	plt.legend()
	plt.savefig('./results/' + 'acc_all.png')
	plt.clf()

# print_acc_all()