# Stein Variational Gradient Descent Bayesian Neural Network

This is a refactor of the SVGD BNN regression implementation from dtak/ocbnn-public. 
The refactor is for the purpose of providing an easy to use extensible Bayesian neural 
network that gives sensible uncertainty estimates. Some improvements from the original
implementation include helper functions to convert data to torch tensors automatically,
and data standardization/normalization for seamless integration with data pipelines. 

Currently, this only supports vanilla BNN regression. Later updates may include classification
and support for OC BNN's. They are not included in the initial implementation due to the 
particular application this will be used for initially.

## Usage

bnn = BNNSVGDRegressor(**config) \
bnn.fit(Xtrain, Ytrain) \
y_mean, y_prediction_interval = bnn.predict(Xtest)

Where Xtrain, Ytrain, and Xtest are numpy arrays (or torch tensors). The config file should follow
the format shown in bnn_config.yaml in examples.