import numpy as np

class Model():
	def __init__(self, input_dim, learning_rate=0.1):
		self.input_dim = input_dim
		self.learning_rate = learning_rate
		self.w = np.random.normal(size=[input_dim,1])

	def forward(self, x):
		"""Allow for batch processing, assume row-major"""
		return self.sigmoid(np.dot(x, self.w))

	def sigmoid(self, x):
		return 1 / (1 + np.exp(-x))

	def error(self, x, y):
		"""Not used in weight update, only for human consumption (and please do so moderately)"""
		"""NOTE as forward(x) approaches 0 or 1, floating point approximationg + log may **** this up"""
		return np.sum(y * np.log(self.forward(x)) + (1-y)*np.log(1-self.forward(x)))

	def gradient(self, x, y):
		return x * (y - self.forward(x))

	def step(self, x, y):
		self.w += self.learning_rate * np.sum(self.gradient(x,y), axis=0).reshape(3,1)