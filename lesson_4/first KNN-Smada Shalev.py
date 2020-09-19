import numpy as np
import matplotlib.pyplot as plt


data = np.array([[6.0, 7.0],
                [2.0, 3.0],
                [3.0, 7.0],
                [4.0, 4.0],
                [5.0, 8.0],
                [6.0, 5.0],
                [7.0, 9.0],
                [8.0, 5.0],
                [8.0, 2.0],
                [10.0, 2.0]])


lbl = np.array([
            [1],
            [1],
            [1],
            [1],
            [2],
            [2],
            [2],
            [2],
            [2]])


categories = np.array([0, 1, 1, 1, 1, 2, 2, 2, 2, 2])
colormap = np.array(['r', 'g', 'b'])
plt.scatter(data[:, 0], data[:, 1], s=100, c=colormap[categories])
plt.show()


def euclidean_distance(p1, p2):
    d = 0.0
    for i in range(len(p1)):
        a = float(p1[i])
        b = float(p2[i])
        d += np.power((a-b), 2)
    d = np.sqrt(d)
    return d


def KNN(train, test, lbl, K):
    distances = []
    for t, l in zip(train, lbl):
        dist = euclidean_distance(test, t)
        distances.append((t, dist, l[0]))
        distances.sort(key=lambda dist: dist[1])
    NN = []
    for i in range(K):
        NN.append(distances[i])
    return NN


def predict(train, test, lbl, K):
    neighbors = KNN(train, test, lbl, K)
    out = [row[-1] for row in neighbors]
    return max(set(out), key=out.count)


a = predict(data[1:], data[0], lbl, 2)

print(a)

