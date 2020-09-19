import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.datasets import fetch_openml
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier
from PIL import Image
import pickle
import os


class ArrayPik:
    def save_list(self, list):
        filename = 'arr.bin'
        outfile = open(filename, 'wb')
        pickle.dump(list, outfile)
        outfile.close()

    def open_list(self):
        filename = 'arr.bin'
        if os.path.isfile(filename):
            infile = open(filename, 'rb')
            array = pickle.load(infile)
            infile.close()
            return array
        array = []
        return array


ddd = ArrayPik()
ar = ddd.open_list()
if not ar:
    datapath = Path("mnist_784.npz")
    if not(datapath.exists()):
        print("downloading file...")
        x_mnist, y_mnist = fetch_openml("mnist_784", version=1,
                return_X_y=True, data_home=".")
        x=np.array(x_mnist, dtype="u8")
        y=np.array(y_mnist, dtype="u8")
        np.savez(datapath, x=x, y=y)
        del x_mnist, y_mnist
    print("Loading file...")
    data = np.load(datapath)
    x_mnist = data["x"]
    x_mnist = x_mnist / 255
    y_mnist = data["y"]


    # plt.figure()
    # for i in range(200):
    #     plt.subplot(10, 20, i+1)
    #     plt.axis("off")
    #     plt.imshow(x_mnist[i].reshape((28, 28)), cmap="gray", vmin=0, vmax=255)
    # plt.show()


    train_data, train_lbl, test_data, test_lbl = x_mnist[:60000], y_mnist[:60000], x_mnist[60000:70000], y_mnist[60000:70000]

    mlp = MLPClassifier(
        hidden_layer_sizes=(100, ),
        max_iter=50,
        solver='sgd',
        verbose=True,
        learning_rate_init=0.1
    )

    mlp.fit(train_data, train_lbl)

    ddd.save_list([mlp, train_data, train_lbl, test_data, test_lbl])
else:
    mlp = ar[0]
    train_data = ar[1]
    train_lbl = ar[2]
    test_data = ar[3]
    test_lbl = ar[4]
    ddd.save_list(ar)
