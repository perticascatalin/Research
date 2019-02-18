from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score
import generator as gen

N_CLASSES = 10
N_ESTIM = 96

def sorting_accuracy(orders_pred, orders_expect):
	acc = 0.0
	for i in range(len(orders_pred)):
		c_acc = 0.0
		for j in range(N_CLASSES):
			predicted = int(orders_pred[i][j])
			actual = orders_expect[i][j]
			#print predicted, actual
			if predicted == actual:
				c_acc += 1.0
		acc += c_acc
	return acc/len(orders_pred)

def tree_model(name):
	if name == "extreme":
		return ExtraTreesClassifier(n_estimators = N_ESTIM)
	elif name == "forest":
		return RandomForestClassifier(n_estimators = N_ESTIM)
	else:
		return tree.DecisionTreeClassifier()

lsts_train, orders_train = gen.get_newer_data()
lsts_val, orders_val = gen.get_newer_data()

clf = tree_model("extreme")
clf = clf.fit(lsts_train, orders_train)

orders_pred = clf.predict(lsts_val)
print sorting_accuracy(orders_pred, orders_val)