from svgd_bnn.svgd import BNNSVGDRegressor
import numpy as np
import matplotlib.pyplot as plt
import yaml
from examples.options_example import BNNOptions


# Load config
file = open("bnn_config.yaml", 'r')
config = yaml.safe_load(file)
file.close()

# Load options
options = BNNOptions(**config)

# Instantiate BNN
bnn = BNNSVGDRegressor(options)


noise = .000001
dim = 1


# Toy function
def f(x):
    return np.cos(x/10)*np.sin(x/7)


# Generate toy data with n data points
def data_gen(n):
    xdata = np.sort(100*np.random.rand(n))
    ydata = f(xdata) + noise*2*np.random.rand(n) - noise

    return xdata.reshape(-1, 1), ydata.reshape(-1, 1)


x, y = data_gen(15)

X = np.linspace(-20, 120, 1000)
Y = f(X)

# Train BNN and make prediction with test data
bnn.fit(x, y)
ypred, ystd = bnn.predict(X)

# Plots
plt.plot(x, y, 'o')
plt.plot(X, Y)
plt.plot(X, ypred)
plt.xlim((-20, 120))
plt.fill_between(X, ypred - ystd, ypred + ystd, alpha=.25)
plt.savefig('../plots/1D_test.png')
plt.show()

