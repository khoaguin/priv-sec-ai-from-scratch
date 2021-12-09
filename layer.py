class Layer:

	def __init__(self):
		self.x = None
		self.y = None
	
	def forward(self, x):
		"""The forward pass - calculates the output 
		y given input x
		"""
		raise NotImplementedError

	def backward(self, dJdy):
		"""Computes dJ/dx given dJ/dy
		"""
		raise NotImplementedError
	
	def update_param(self, lr):
		"""[summary]

		Args:
			lr (float): the learning rate
		"""
		raise NotImplementedError

	