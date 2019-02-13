from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import generator as gen

N_CLASSES = 10

lsts_train, orders_train = gen.get_newer_data()
lsts_val, orders_val = gen.get_newer_data()

#clf = tree.DecisionTreeClassifier()
clf = RandomForestClassifier(n_estimators = 72)
clf = clf.fit(lsts_train, orders_train)

orders_pred = clf.predict(lsts_val)
#print orders_val
#print orders_pred

acc = 0.0
for i in range(len(orders_pred)):
	c_acc = 0.0
	for j in range(N_CLASSES):
		predicted = int(orders_pred[i][j])
		actual = orders_val[i][j]
		#print predicted, actual
		if predicted == actual:
			c_acc += 1.0
	acc += c_acc
print "accuracy ", acc/len(orders_pred)