import numpy as np 
from termcolor import colored
 
class NeuralNetwork:
	def __init__(self,inputLayerNeurons, hiddenLayerNeurons, outputLayerNeurons):
		self.hidden_weights = np.random.uniform(size=(inputLayerNeurons,hiddenLayerNeurons))
		self.hidden_bias =np.random.uniform(size=(1,hiddenLayerNeurons))
		self.output_weights = np.random.uniform(size=(hiddenLayerNeurons,outputLayerNeurons))
		self.output_bias = np.random.uniform(size=(1,outputLayerNeurons))
		self.hidden_layer_output = np.random.uniform(size=(1,outputLayerNeurons))
 
	def sigmoid(self,x):
		return 1.0/(1.0 + np.exp(-x))
 
	def sigmoid_derivative(self,x):
		return x * (1.0 - x)
	
	def predict(self, inpt):
		#Forward Propagation
		hidden_layer_activation = np.dot(inpt,self.hidden_weights)
		hidden_layer_activation += self.hidden_bias
		self.hidden_layer_output = self.sigmoid(hidden_layer_activation)

		output_layer_activation = np.dot(self.hidden_layer_output,self.output_weights)
		output_layer_activation += self.output_bias
		return self.sigmoid(output_layer_activation)

	def train(self, inpt, exp_out, learningRate=0.1, epochs=10000):
		for _ in range(epochs):
			#Forward Propagation
			predicted_output = self.predict(inpt)

			#Back Propagation
			error = exp_out - predicted_output
			
			d_predicted_output = error * self.sigmoid_derivative(predicted_output)		
			error_hidden_layer = d_predicted_output.dot(self.output_weights.T)
			d_hidden_layer = error_hidden_layer * self.sigmoid_derivative(self.hidden_layer_output)
 
			#Updating Weights and Biases
			self.output_weights += self.hidden_layer_output.T.dot(d_predicted_output) * learningRate
			self.output_bias += np.sum(d_predicted_output,axis=0,keepdims=True) * learningRate
			self.hidden_weights += inpt.T.dot(d_hidden_layer) * learningRate
			self.hidden_bias += np.sum(d_hidden_layer,axis=0,keepdims=True) * learningRate
 
#Input datasets
inputs = np.array([[0,0],[0,1],[1,0],[1,1]])
expected_output = np.array([[0],[1],[1],[0]])
nn = NeuralNetwork(2,2,1)
 
print("\nInitial hidden weights:\n",colored(nn.hidden_weights, 'red'))
print("\nInitial hidden biases:\n",colored(nn.hidden_bias, 'red'))
print("\nInitial output weights:\n",colored(nn.output_weights, 'red'))
print("\nInitial output biases:\n",colored(nn.output_bias, 'red'))
 
nn.train(inputs, expected_output)
 
print("\nFinal output bias:\n",colored(nn.hidden_weights, 'green'))
print("\nFinal output bias:\n",colored(nn.hidden_bias, 'green'))
print("\nFinal output bias:\n",colored(nn.output_weights, 'green'))
print("\nFinal output bias:\n",colored(nn.output_bias, 'green'))

test = nn.predict(inputs)
print("\nOutput from neural network after 10,000 epochs:\n",colored(test, 'blue'))