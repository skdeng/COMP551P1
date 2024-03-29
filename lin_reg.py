import numpy as np

class Model(object):
	def __init__(self, input_dim):
		self.input_dim = input_dim
		self.w = np.random.normal(size=[input_dim+1,1])

	def solve(self, x, y):
		""" A little bit different from the equations on the slides because we are using row major matrices, i.e. each row is a data point"""
		x = np.concatenate((np.ones([x.shape[0], 1]), x), axis=1)
		xtx = np.dot(x.T, x)
		xty = np.dot(y, x)
		self.w = np.dot(np.linalg.inv(xtx), xty.T)

	def forward(self, x):
		x = np.concatenate((np.ones([x.shape[0], 1]), x), axis=1)		
		return np.dot(x, self.w)

	def error(self, x, y):
		return np.square(y - self.forward(x))
