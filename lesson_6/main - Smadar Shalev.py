from sort_numbers_fit_SmadarShalev import *


def is_num(url, to_print):
    to_print = False
    x = Image.open(url)
    x = x.resize((28, 28))
    x = x.convert('L')
    x = np.array(x)
    if to_print:
        x = x.reshape((1, 784))
        plt.axis("off")
        plt.imshow(x.reshape((28, 28)), cmap="gray", vmin=0, vmax=255)
        plt.show()
        x = x / 255
    else:
        x = x / 255
        x = x.reshape((1, 784))
    return mlp.predict(x)


# print("Training set score: %f" % mlp.score(train_data, train_lbl))
# print("Test set score: %f" % mlp.score(test_data, test_lbl))

# lbl_pred = is_num("0.png", True)
# print(lbl_pred)
# lbl_pred = is_num("1.png", True)
# print(lbl_pred)
# lbl_pred = is_num("2.png", True)
# print(lbl_pred)
# lbl_pred = is_num("3.png", True)
# print(lbl_pred)
# lbl_pred = is_num("4.png", True)
# print(lbl_pred)
# lbl_pred = is_num("5.png", True)
# print(lbl_pred)
# lbl_pred = is_num("6.png", True)
# print(lbl_pred)
# lbl_pred = is_num("7.png", True)
# print(lbl_pred)
# lbl_pred = is_num("8.png", True)
# print(lbl_pred)
# lbl_pred = is_num("9.png", True)
# print(lbl_pred)
lbl_pred = is_num("try.png", True)
print(lbl_pred[0])

# print(confusion_matrix(test_lbl, mlp.predict(test_data)))
