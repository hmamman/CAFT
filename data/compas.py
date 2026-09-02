import numpy as np
from src.utils.config import compas
import sys
sys.path.append("../")


def compas_data():
    X = []
    Y = []
    i = 0

    with open("datasets/compas", "r") as ins:
        for line in ins:
            line = line.strip()
            line1 = line.split(',')
            if i == 0:
                i += 1
                continue
            X.append(line1[:-1])
            if int(line1[-1]) == 0:
                Y.append(0)
            else:
                Y.append(1)
        X = np.array(X, dtype=float)
        Y = np.array(Y, dtype=int).ravel()  # Flatten Y to a 1D array

    input_shape = (None, compas.params)
    nb_classes = 2
    return X, Y, input_shape, nb_classes