import tensorflow as tf 
from tensorflow.keras import layers, models, optimizers, callbacks


class NeuralNet(object):

	def __init__(self, batch_size, max_step_size):
		self.batch_size = batch_size
		self.max_step_size = max_step_size

	# Create graph with convolutional, pooling and fully connected layers
	def generateModel(self, numOutputs = [48,64,128, 256], kernelSize = (5,5), filterPadding = 'same'):
		imageSize = (120, 100, 1)
		inputImage = layers.Input(shape=imageSize)

		convLayers = self.generateConvLayers(inputImage, numOutputs, kernelSize, filterPadding)
		flattenLayers = self.flattenCNNOutput(convLayers)
		denseLayers = self.generateDenseLayers(flattenLayers)

		outputReshaped =  tf.reshape(denseLayers, [-1, 4, 10])

		model = models.Model(inputImage, outputReshaped)


		return model
	

	# Build Convolutional Layers in TF Graph
	def generateConvLayers(self, inputImage, numOutputs, kernelSize, filterPadding):

		# Input layer
		convLayers = layers.Conv2D(filters = numOutputs[0], kernel_size = kernelSize, padding = filterPadding, strides = (2,2))(inputImage)
		convLayers = layers.Activation('relu')(convLayers)
		convLayers = self.dropoutLayer(convLayers)
		convLayers = self.maxPool(convLayers, True)

		# Second CNN Layer
		convLayers = layers.Conv2D(filters = numOutputs[1], kernel_size = kernelSize, padding = filterPadding)(convLayers)
		convLayers = layers.Activation('relu')(convLayers)
		convLayers = self.dropoutLayer(convLayers)
		convLayers = self.maxPool(convLayers, False)

		# Third CNN Layer
		convLayers = layers.Conv2D(filters = numOutputs[2], kernel_size = kernelSize, padding = filterPadding)(convLayers)
		convLayers = layers.Activation('relu')(convLayers)
		convLayers = self.dropoutLayer(convLayers)
		convLayers = self.maxPool(convLayers, True)

		convLayers = layers.Conv2D(filters = numOutputs[3], kernel_size = kernelSize, padding = filterPadding)(convLayers)
		convLayers = layers.Activation('relu')(convLayers)
		convLayers = self.dropoutLayer(convLayers)
		convLayers = self.maxPool(convLayers, False)


		return convLayers
	

	# 2x2 Max Pooling
	def maxPool(self, layer, stride):
		if(not stride):
			return layers.MaxPooling2D()(layer)

		return layers.MaxPooling2D(strides=(2,2))(layer)

	# Reduce overfitting (?)
	def dropoutLayer(self, layer):
		return layers.Dropout(rate=0.2)(layer)

	# Flatten 4D Tensor so it can be connected to dense network
	def flattenCNNOutput(self, layer):
		return layers.Flatten()(layer)

	# Connect pooled convolutional layers to a fully-connected neural net with softmax function
	def generateDenseLayers(self, layer):
		# First fully connected network
		dense = layers.Dense(1024)(layer)
		dense = self.dropoutLayer(dense)
		dense = layers.Activation('relu')(dense)

		# Second layer of fully connected network
		# Output layer
		dense = layers.Dense(40)(dense)
		dense = layers.Activation('relu')(dense)
		outputImage = layers.Activation('softmax')(dense)


		return dense




