from svgd_bnn.svgd import BNNSVGDRegressor
import numpy as np
import matplotlib.pyplot as plt
import yaml


file = open("bnn_config.yaml", 'r')
config = yaml.safe_load(file)
file.close()


bnn = BNNSVGDRegressor(**config)


noise = .000001
dim = 1

def f(x):
    return np.cos(x/10)*np.sin(x/7)


def data_gen(n):
    xdata = np.sort(100*np.random.rand(n))
    ydata = f(xdata) + noise*2*np.random.rand(n) - noise

    return xdata.reshape(-1, 1), ydata.reshape(-1, 1)

x, y = data_gen(15)

X = np.linspace(0, 100, 1000)
Y = f(X)

bnn.fit(x, y)
ypred, ystd = bnn.predict(X)

plt.plot(x, y, 'o')
plt.plot(X, Y)
plt.plot(X, ypred)
plt.fill_between(X, ypred - ystd, ypred + ystd, alpha=.25)

plt.show()

