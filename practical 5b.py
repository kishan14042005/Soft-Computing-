import numpy as np
from scipy.linalg import norm, pinv
from matplotlib import pyplot as plt

class RBF:

    def __init__(self, indim, numCenters, outdim):
        self.indim = indim
        self.outdim = outdim
        self.numCenters = numCenters
        self.centers = [np.random.uniform(-1, 1, indim) for _ in range(numCenters)]
        self.beta = 8
        self.W = np.random.random((self.numCenters, self.outdim))

    def _basisfunc(self, c, d):
        assert len(d) == self.indim
        return np.exp(-self.beta * norm(c - d)**2)

    def _calcAct(self, X):
        # calculate activations of RBFs
        G = np.zeros((X.shape[0], self.numCenters), float)
        for ci, c in enumerate(self.centers):
            for xi, x in enumerate(X):
                G[xi, ci] = self._basisfunc(c, x)
        return G

    def train(self, X, Y):
        """ X: matrix n × indim
            Y: matrix n × outdim """

        # choose random centers from training data
        rnd_idx = np.random.permutation(X.shape[0])[:self.numCenters]
        self.centers = [X[i, :] for i in rnd_idx]

        print("Centers:", self.centers)

        # calculate activations
        G = self._calcAct(X)

        # compute weights via pseudoinverse
        self.W = np.dot(pinv(G), Y)

    def test(self, X):
        G = self._calcAct(X)
        Y = np.dot(G, self.W)
        return Y


if __name__ == '__main__':

    # ----- 1D Example -----
    n = 100
    x = np.linspace(-1, 1, n).reshape(n, 1)

    # target function
    y = np.sin(3*(x + 0.5)**3 - 1)

    # rbf regression
    rbf = RBF(1, 10, 1)
    rbf.train(x, y)
    z = rbf.test(x)

    # plot original data
    plt.figure(figsize=(12, 8))
    plt.plot(x, y, 'k-', label="Original")

    # learned model
    plt.plot(x, z, 'r-', linewidth=2, label="RBF Approx")

    # plot RBF centers
    plt.plot(rbf.centers, np.zeros(rbf.numCenters), 'gs', label="Centers")

    # individual RBF plots
    for c in rbf.centers:
        cx = np.arange(float(c) - 0.7, float(c) + 0.7, 0.01)
        cy = [rbf._basisfunc(np.array([c]), np.array([cx_])) for cx_ in cx]
        plt.plot(cx, cy, '-', color='gray', linewidth=0.3)

    plt.xlim(-1.2, 1.2)
    plt.legend()
    plt.show()
