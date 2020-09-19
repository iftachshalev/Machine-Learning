import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
import sklearn.neural_network


vir_iris_data = np.genfromtxt('iris_flowers .csv', delimiter=',')
random_iris_data = np.random.permutation(vir_iris_data)
test_data = random_iris_data[0:20, :4]
train_data = random_iris_data[20:, :4]
test_lbl = random_iris_data[0:20, 4:]
train_lbl = random_iris_data[20:, 4:]


def euclidean_distance(p1, p2):
    d = 0.0
    for i in range(len(p1)):
        a = float(p1[i])
        b = float(p2[i])
        d += np.power((a-b),2)
    d = np.sqrt(d)
    return d


def takeSecond(elem):
    return elem[1]


def predict(train, test, lbl, K):
    distances = []
    for t, l in zip(train, lbl):
        dist = euclidean_distance(test, t)
        distances.append([t, dist, l[0]])
    distances.sort(key=takeSecond)
    neighbors = []
    for i in range(K):
        neighbors.append(distances[i])
    out = [row[-1] for row in neighbors]
    return max(out, key=out.count)


for i in range(len(test_data)):
    result = predict(train_data, test_data[i], train_lbl, 3)
    print(test_lbl[i], result)


mlp = sklearn.neural_network.MLPClassifier(
    hidden_layer_sizes=(5, 10, 33, 3),
    solver='sgd',
    learning_rate_init=0.01,
    max_iter=5000)
mlp.fit(train_data, train_lbl)
lbl = mlp.predict(test_data)
print(confusion_matrix(test_lbl, lbl))


classifier = KNeighborsClassifier(n_neighbors=3)
classifier.fit(train_data, train_lbl)
lbl_pred = classifier.predict(test_data)
for t, p in zip(test_lbl, lbl_pred):
    if t == p:
        print(1)
    else:
        print(0)
print(confusion_matrix(test_lbl, lbl_pred))
