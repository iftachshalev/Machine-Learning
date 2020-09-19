import numpy as np 
from termcolor import colored


class Perceptron:

	def __init__(self, inputLayerNeurons, outputLayerNeurons):
		self.output_weights = np.random.uniform(size=(inputLayerNeurons,outputLayerNeurons))
		self.output_bias = np.random.uniform(size=(1,outputLayerNeurons))

	def sigmoid(self,x):
		return 1.0/(1.0 + np.exp(-x))

	def sigmoid_derivative(self, x):
		return x * (1.0 - x)
	
	def predict(self, input):
		#Forward Propagation
		output_layer_activation = np.dot(input,self.output_weights)
		output_layer_activation += self.output_bias
		return self.sigmoid(output_layer_activation)

	def train(self, inpt, exp_out, learningRate=0.01, epochs=20000):
		for _ in range(epochs):
			# Forward Propagation
			predicted_output = self.predict(inpt)

			# Back Propagation
			error = exp_out - predicted_output
			d_predicted_output = error * self.sigmoid_derivative(predicted_output)

			# Updating Weights and Biases
			self.output_weights += inpt.T.dot(d_predicted_output) * learningRate
			self.output_bias += np.sum(d_predicted_output,axis=0,keepdims=True) * learningRate


# Input datasets
inputs = np.array([[0,0],[0,1],[1,0],[1,1]])
expected_output = np.array([[0],[1],[0],[0]])

nn = Perceptron(2,1)
print("\nInitial output weights:\n",colored(nn.output_weights, 'red'))
print("\nInitial output biases:\n",colored(nn.output_bias, 'red'))
 
nn.train(inputs, expected_output)
print("\nFinal output bias:\n",colored(nn.output_weights, 'green'))
print("\nFinal output bias:\n",colored(nn.output_bias, 'green'))

test = nn.predict(inputs)
print("\nOutput from neural network after 20,000 epochs:\n",colored(test, 'blue'))