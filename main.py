import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def predict(x_p, args, labels):
    preds = np.empty(0)
    classes_cnt = labels.shape[1]
    n = labels.shape[0]
    for i in range(classes_cnt):
        cnt_y_i = np.count_nonzero(labels[:, i])
        x_given_y_k = ((np.count_nonzero(np.multiply(args, labels[:, i].reshape(n, 1)) == x_p, axis=0)) + 1) \
            / (cnt_y_i + classes_cnt)
        preds = np.append(preds, np.prod(x_given_y_k) * (cnt_y_i / n))
    return preds


def pred_0_1(x_p, args, labels):
    a = predict(x_p, args, labels)
    if a[0] < a[1]:
        return 1
    else:
        return 0

def train(x, y, k):
    discretization_constant = 1
    n = len(y)
    labels = np.zeros((len(y), k))
    args = np.zeros(x.shape)
    for i in range(n):
        labels[i, y[i]] = 1
        for j in range(x.shape[1]):
            args[i, j] = x[i, j] // discretization_constant

    err = 0
    for i in range(n):
        a = pred_0_1(x[i, :].reshape(1, x.shape[1])//discretization_constant, args, labels)
        if y[i] != a:
            color = '#01ff01'
            err += 1
        elif a == 1:
            color = '#ff2200'
        else:
            color = '#1f77b4'
        plt.scatter(x[i, 0], x[i, 1], c=color)

    plt.show()
    print(err)



if __name__ == '__main__':
    data = pd.read_csv('data.csv').dropna()
    x = data.iloc[:, :-1].to_numpy()
    y = data.iloc[:, -1].to_numpy()
    train(x, y, 2)
