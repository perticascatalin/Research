import random
import time

# Returns unique indexes randomly picked from A
def pick_rand_k(A, k):
	lst = []
	for i in range(k):
		while True:
			num = random.randint(0, len(A) - 1)
			# Condition to generate unique numbers
			if lst.count(num) == 0:
				lst.append(num)
				break
			else:
				continue
	return lst

# Get actual values of indexes
def get_k_vals(A, lst):
	vals = []
	for i in lst:
		vals.append(A[i])
	return vals

# Check if array sorted
def sorted(A):
	B = list(A)
	B.sort()
	for p in zip(A,B):
		if p[0] != p[1]:
			return False
	return True

# Sort rand k algorithm
def sort_rand_k(A, k):
	while sorted(A) != True:
		print (A)
		k_lst = pick_rand_k(A, k)
		k_lst.sort()
		#print k_lst
		k_vals = get_k_vals(A, k_lst)
		k_vals.sort()
		#print k_vals

		for i in range(len(k_lst)):
			j = k_lst[i]
			A[j] = k_vals[i]
		time.sleep(1)


A = [23, 8, 2, 9, 10, 11, 48, 28, 29, 49, 15, 16, 17, 18]
sort_rand_k(A, 8)
print A