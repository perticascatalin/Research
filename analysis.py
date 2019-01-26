import tensorflow as tf
import numpy as np

def pretty_printing(logits, y_exp, x):
	# this requires special print
	#print logits[0]

	# todo: migrate printing to an analysis module 
	# which stores interesting predictions for your study
	out = list()
	y_pred = list()
	for j in range(10):
		f_max = np.argmin(logits[0][j])
		f_max_2 = np.argmin(logits[0][j])
		for k in range(10):
			if logits[0][j][k] > logits[0][j][f_max]:
				f_max_2 = f_max
				f_max = k
			elif logits[0][j][k] > logits[0][j][f_max_2]:
				f_max_2 = k
		y_pred.append(f_max)
		out.append((f_max, f_max_2))
	print x[0]
	print y_pred
	print y_exp[0]