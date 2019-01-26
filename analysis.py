import tensorflow as tf
import numpy as np

def debugger(correct_pred, logits, y_exp, x):
	print correct_pred[0]
	# print len(logits)
	# print logits[0]
	# print len(logits[0])
	out = list()
	for j in range(10):
		f_max = np.argmin(logits[0][j])
		f_max_2 = np.argmin(logits[0][j])
		for k in range(10):
			if logits[j][0][k] > logits[j][0][f_max]:
				f_max_2 = f_max
				f_max = k
			elif logits[j][0][k] > logits[j][0][f_max_2]:
				f_max_2 = k
		out.append((f_max, f_max_2))
	#print out


def pretty_printing(correct_pred, logits, y_exp, x):
	out = list()
	y_pred = list()
	for j in range(10):
		y_pred.append(np.argmax(logits[j][0]))
	print x[0]
	print y_pred
	print y_exp[0]