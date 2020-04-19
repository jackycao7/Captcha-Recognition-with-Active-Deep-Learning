import tensorflow as tf 
from tensorflow.keras import callbacks, backend
import math

class ChangeLearningRate(callbacks.Callback):
	def __init__(self):
		self.a = 0.01
		self.B = 0.75
		self.y = 0.0001
		self.u = 0.9
		self.iter = 0

	def on_train_batch_begin(self, batch, logs=None):
		currLR = float(backend.get_value(self.model.optimizer.lr))
		inner = 1 + (self.y * self.iter)
		newLR = self.a * math.pow(inner, self.B)
		backend.set_value(self.model.optimizer.lr, newLR)
		self.iter += 100
		#print(newLR, float(backend.get_value(self.model.optimizer.lr)))
		
