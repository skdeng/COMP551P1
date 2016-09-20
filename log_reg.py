import numpy as np

class Model():
	def __init__(self, input_dim, learning_rate=0.1):
		self.input_dim = input_dim
		self.learning_rate = learning_rate
		self.w = np.random.normal(size=[input_dim,1])

	def forward(self, x):
		"""Allow for batch processing, assume row-major"""
		return np.round(self.sigmoid(np.dot(x, self.w)))

	def sigmoid(self, x):
		return 1 / (1 + np.exp(-x))

	def error(self, x, y):
		"""Not used in weight update, only for human consumption (and please do so moderately)"""
		"""NOTE as forward(x) approaches 0 or 1, floating point approximationg + log may **** this up"""
		# return np.sum(y * np.log(np.maximum(self.forward(x),[0.000001])) + (1-y)*np.log(np.maximum(1-self.forward(x), [0.0000001]))) / x.shape[1]
		y = y.reshape(y.shape[0],1).T
		return np.dot(y, np.log(self.forward(x) + 0.000001)) + np.dot((1-y), np.log(1-self.forward(x) + 0.000001)) / x.shape[0]

	def gradient(self, x, y):
		# print x.shape, (y-self.forward(x)).shape, self.w.shape
		return np.dot((y - self.forward(x)), x)

	def step(self, x, y, e=float("inf")):
		"""By default, this function performs one update step
			error epsilon can be changed to perform multi updates at once"""
		self.w += self.learning_rate * np.sum(self.gradient(x,y), axis=0).T.reshape(self.input_dim, 1)

	def save(self, filename):
		np.save(filename, self.w, allow_pickle=False)

	def load(self, filename):
		np.load(filename, self.w, allow_pickle=False)
