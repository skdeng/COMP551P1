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

# OR function
x = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
y = np.array([[0],[1],[1],[1]], dtype=float)
# naive bayes should be able to learn this
m.fit(x,y,True)
print m.ty1, m.tx1, m.tx0
print m.forward(x)