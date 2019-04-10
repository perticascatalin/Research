import math
import numpy as np
import matplotlib.pyplot as plt

def lin(x):
	return math.log(x, 2)
	return x

# returns array of possible arrangements
def arrangement(maxint, n_size):
	start = 1
	arr = [lin(1)]
	for i in range(n_size):
		start *= (maxint - i)
		arr.append(lin(start))
	return arr

# returns array of factorials
def factorial(n_size):
	start = 1
	arr = [lin(1)]
	for i in range(n_size):
		start *= (i+1)
		arr.append(lin(start))
	return arr

# returns array of pow2s
def pow2(n_size):
	start = 1
	arr = [lin(1)]
	for i in range(n_size):
		start *= 2
		arr.append(lin(start))
	return arr

# returns approximate 2^(n^2)
def pow2comb(n_size):
	start = 1
	arr = [lin(1)]
	for i in range(n_size):
		n = (i+1)
		p = (n * (n-1))/2
		tn = pow2(p)
		start = tn[-1]
		arr.append(start)
	return arr

n = 20

index = np.arange(n+1)
arang = arrangement(50, n)
fact = factorial(n)
p2 = pow2(n)
p2c = pow2comb(n)

print index
print arang
print fact
print p2c
print p2

figname = 'pc.png'
plt.title('Problem Complexity', fontsize = 18)
plt.xlabel('N', fontsize = 18)
plt.ylabel('log(N)', fontsize = 18)
plt.plot(index, arang, 'm', linewidth = 2.0, label = 'Arrangements')
plt.plot(index, fact, 'r', linewidth = 2.0, label = 'Permutations')
plt.plot(index, p2, 'g', linewidth = 2.0, label = 'Exponential')
plt.plot(index, p2c, 'b', linewidth = 2.0, label = 'Combinations')
plt.legend(loc = 'upper left')
#plt.show()
plt.savefig('./results/' + figname)
plt.clf()
