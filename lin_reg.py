import numpy as np

class Model(object):
	def __init__(self, input_dim, learning_rate=0.1):
		self.input_dim = input_dim
		self.learning_rate = learning_rate
		self.w = np.random.normal(size=[input_dim,1])

	def solve(self, x, y):
		xtx = np.dot(x.T, x)
		xty = np.dot(y, x)
		self.w = np.dot(np.linalg.inv(xtx), xty.T)

	def forward(self, x):
		return np.dot(x, self.w)