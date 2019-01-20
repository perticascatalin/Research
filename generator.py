import random

LST_SIZE = 10
lst = []
order = []

for i in range(LST_SIZE):
	lst.append(random.random())

for i in range(LST_SIZE):
	count = 0
	for j in range(LST_SIZE):
		if lst[j] < lst[i]:
			count += 1
	order.append(count)

print lst
print order