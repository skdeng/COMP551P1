import numpy as np

class Model():
	def __init__(self, input_dim):
		self.input_dim = input_dim

	def fit(self, x, y, laplace_smoothing=True):
		self.ty1 = np.sum(y, axis=0) / y.size

		smoothu = 1 if laplace_smoothing else 0
		smoothv = 2 if laplace_smoothing else 0

		y1 = np.where(y==1)[0]
		self.tx1 = (np.sum(x[y1], axis=0) + smoothu) / (y1.size + smoothv)
		
		y0 = np.where(y==0)[0]
		self.tx0 = (np.sum(x[y0], axis=0) + smoothu) / (y0.size + smoothv)

	def forward(self, x):
		# P(x|y=1)
		pxy = x * self.tx1
		pyx = pxy * self.ty1
		py = np.sum(pxy, axis=1)
		pxy0 = x * self.tx0
		py0x = pxy0 * (1 - self.ty1)
		py0 = np.sum(pxy0, axis=1)
		return (py > py0) * 1	# equivalent to computing the ratio