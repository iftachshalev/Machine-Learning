import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


num_of_data = 11


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


fig = plt.figure()
ax = fig.add_subplot(111)

sea_array = []
sea_array_2 = []
for i in range(num_of_data):
    img = Image.open("IMG//sea"+str(i)+".jpg")
    img.load()
    sea = np.array(img, dtype=np.uint8)
    sea_array.append(sea[:, :, 1].sum()/sea[:, :, 1].size)
    sea_array_2.append(sea[:, :, 2].sum()/sea[:, :, 2].size)
print(sea_array)

land_array = []
land_array_2 = []
for i in range(num_of_data):
    img = Image.open("IMG//land"+str(i)+".jpg")
    img.load()

    land = np.array(img, dtype=np.uint8)
    land_array.append(land[:, :, 1].sum()/land[:, :, 1].size)
    land_array_2.append(land[:, :, 2].sum()/land[:, :, 2].size)
print(land_array)

for i in range(num_of_data):
    plt.plot(sea_array_2[i], sea_array[i], 'b^')
    plt.plot(land_array_2[i], land_array[i], 'g*')

real_array = list()
true_array = list()
for i in range(num_of_data):
    real_array.append([land_array[i], land_array_2[i]])
    true_array.append(1)
    real_array.append([sea_array[i], sea_array_2[i]])
    true_array.append(0)


perceptron = Perceptron(2, 10000, 0.01)
b = perceptron.bios
w = perceptron.weights
inputs = np.array(real_array)
labels = np.array(true_array)

perceptron.train(inputs, labels)


for i in range(num_of_data):
    ar = np.array([[land_array[i], land_array_2[i]]])
    print(ar, perceptron.predict(ar))
    ar = np.array([[sea_array[i], sea_array_2[i]]])
    print(ar, perceptron.predict(ar))

plt.show()