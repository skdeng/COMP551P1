import numpy as np

import naive_bayes

m = naive_bayes.Model(2)

# XOR function
x = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
y = np.array([[0],[1],[1],[0]], dtype=float)
# naive bayes should not be able to learn XOR
m.fit(x,y,True)
print m.ty1, m.tx1, m.tx0
print m.forward(x)
print 

# OR function
x = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
y = np.array([[0],[1],[1],[1]], dtype=float)
# naive bayes should be able to learn this
m.fit(x,y,True)
print m.ty1, m.tx1, m.tx0
print m.forward(x)
print

x = np.zeros([16,4])
for i in xrange(16):
	x[i][3] = i % 2
	x[i][2] = i in [2,3,6,7,10,11,14,15]
	x[i][1] = i in [4,5,6,7,12,13,14,15]
	x[i][0] = i >= 8
y = np.zeros([16,1])
for i in xrange(16):
	if x[i][0] == 1:
		y[i] = 1
m.fit(x,y, False)
print m.ty1, m.tx1, m.tx0
print m.forward(x)