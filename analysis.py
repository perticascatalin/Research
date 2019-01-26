import tensorflow as tf

def pretty_printing(logits, y_exp, x):
	# this requires special print
	#print logits[0]

	# todo: migrate printing to an analysis module 
	# which stores interesting predictions for your study
	out = list()
	for j in range(10):
		f_max = 0
		f_max_2 = 0
		for k in range(10):
			if logits[0][j][k] > logits[0][j][f_max]:
				f_max_2 = f_max
				f_max = k
			elif logits[0][j][k] > logits[0][j][f_max_2]:
				f_max_2 = k
		out.append((f_max, f_max_2))
	print out
	print y_exp[0]
	print x[0]