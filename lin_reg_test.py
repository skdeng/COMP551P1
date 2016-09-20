import lin_reg
import numpy as np
import matplotlib.pyplot as plt

x = np.array(range(1, 21))
noise = np.random.normal(size=[1,20])
y = noise+x

plt.scatter(x, y)
x = x.reshape(20,1)
m = lin_reg.Model(1)

m.solve(x,y)
plt.plot(x, m.forward(x))
plt.show()