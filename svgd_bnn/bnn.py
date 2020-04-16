import numpy as np
import torch
import os

from torch.autograd import Variable
from torch.distributions.normal import Normal
from torch.distributions.uniform import Uniform
from torch.distributions.dirichlet import Dirichlet
from torch.distributions.multivariate_normal import MultivariateNormal as MVN
import logging


class BayesianNeuralNetwork:
    def _unpack_layers(self, weights):
        """ Helper function for forward pass. Code taken from PyTorch.
        This currently only works for feedforward NN

        """
        num_weight_samples = weights.shape[0]
        for m, n in self.layer_shapes:
            yield weights[:, :m * n].reshape((num_weight_samples, m, n)), weights[:, m * n:m * n + n].reshape(
                (num_weight_samples, 1, n))
            weights = weights[:, (m + 1) * n:]

    def _nonlinearity(self, x):
        """ Activation function. """
        if self.activation == "rbf":
            return torch.exp(-(x).pow(2))  # gaussian rbf epsilon = 1
        elif self.activation == 'tanh':
            return torch.tanh(x)
        return x

    def forward(self, X, weights=None):
        """ Forward pass of BNN. Code taken from PyTorch. """
        if weights is None:
            weights = self.weights

        if weights.ndimension() == 1:
            weights = weights.unsqueeze(0)

        num_weight_samples = weights.shape[0]
        X = X.expand(num_weight_samples, *X.shape)

        for W, b in self._unpack_layers(weights):
            outputs = torch.einsum('mnd,mdo->mno', [X, W]) + b
            X = self._nonlinearity(outputs)

        outputs = outputs.squeeze(dim=0)
        return outputs

    def log_weight_prior(self, weights=None):
        """ Computes the "standard" prior, i.e. Gaussian log-probability of BNN weights. """
        if weights is None:
            weights = self.weights
        return self.weight_dist.log_prob(weights).sum()

    def log_prior(self):
        """ Computes the log-prior term. """
        prior = self.log_weight_prior()
        return prior

    def log_posterior(self, batch_indices=None):
        """ Computes the log-posterior term. """
        return self.log_prior() + self.log_likelihood(batch_indices=batch_indices)


class BNNRegressor(BayesianNeuralNetwork):
    def __init__(self):
        super().__init__()

        # Initialize all weights.
        self.xdim = self.layers[0]
        self.ydim = self.layers[-1]

        self.layer_shapes = list(zip(self.layers[:-1], self.layers[1:]))
        self.n_weights = sum((m + 1) * n for m, n in self.layer_shapes)
        self.weight_dist = Normal(0, self.sigma_w)
        self.noise_dist = Normal(0, self.sigma_noise)
        self.weights = Variable(self.weight_dist.sample(torch.Size([self.n_weights])), requires_grad=True)

    def log_likelihood(self, batch_indices=None):
        """ Computes log-likelihood term. """
        if batch_indices is None:
            batch = self.X_train
            target = self.Y_train
            multiplier = 1
        else:
            batch = self.x_train[batch_indices]
            target = self.y_train[batch_indices]
            multiplier = (self.N_train / len(batch_indices))
        means = self.forward(X=batch)
        if self.Ydim == 1:
            return multiplier * self.noise_dist.log_prob(means - target).sum()
        return multiplier * MVN(means, self.sigma_noise * torch.eye(self.ydim)).log_prob(target).sum()



