import log_reg
import numpy as np

m = log_reg.Model(3, 0.1)

# generate 50 batches of 10 data points each
xset = [np.random.randint(-10,10, size=(10,3)) for i in xrange(50)]

yhelp = [np.sum(xset[i], axis=1) for i in xrange (len(xset))]
yset = [((yhelp[i] > [0] * 10) * 1).reshape(10,1) for i in xrange(len(yhelp))]

for i in xrange(len(xset)):
	m.step(xset[i], yset[i])
	print m.error(xset[i], yset[i])

print m.forward([[50,2,3], [-1,0,0], [-50,-50,0]])