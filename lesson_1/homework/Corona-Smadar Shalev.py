import numpy as np


class Perceptron(object):
    def __init__(self, numOfInputs, epochs=200, learningRate=0.01):
        self.epochs = epochs
        self.learningRate = learningRate
        self.weights = np.zeros(numOfInputs)
        self.bios = 1

    def Activation(self, s):
        if s > 0:
            activation = 1
        else:
            activation = 0
        return activation

    def predict(self, inputs):
        sum = np.dot(inputs, self.weights) + self.bios
        out = self.Activation(sum)
        return out

    def train(self, inputs, labels):
        for _ in range(self.epochs):
            for i in range(len(inputs)):
                prd = self.predict(inputs[i])
                self.weights -= (prd - labels[i]) * inputs[i] * self.learningRate
                self.bios -= (prd - labels[i]) * self.learningRate


inputs = np.array([[1, 0, 0, 1], [1, 0, 0, 0], [0, 0, 1, 1], [0, 1, 0, 0], [1, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [1, 1, 1, 1], [0, 0, 0, 0]])
labels = np.array([1, 1, 0, 0, 1, 0, 0, 1, 0])
perceptron = Perceptron(4)
perceptron.train(inputs, labels)
while True:
    smell = int(input("Loss of smell (1=yes 0=no): "))
    body = int(input("Body heat about 38 (1=yes 0=no): "))
    runny = int(input("Runny nose (1=yes 0=no): "))
    pain = int(input("Body pain (1=yes 0=no): "))
    exit = int(input("Exit (1=yes 0=no): "))
    if exit:
        break
    else:
        t1 = np.array([smell, body, runny, pain])
        if perceptron.predict(t1):
            print("Corona Virus Positive!!!!")
        else:
            print("Corona Virus Negative!!!!")
