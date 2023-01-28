import os

def make_dirs(model_name):
	checkpts_dir = './data/checkpts/' + model_name + '/'
	stats_dir = './data/stats/' + model_name + '/'
	results_dir = './data/results/' + model_name + '/'
	labels_dir = results_dir + 'labels/'
	dirs = [checkpts_dir, stats_dir, results_dir, labels_dir]
	for direct in dirs:
		if not os.path.exists(direct):
			os.makedirs(direct)

	return checkpts_dir, stats_dir, results_dir, labels_dir