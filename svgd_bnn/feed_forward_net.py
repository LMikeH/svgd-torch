
import numpy as np
import torch

from torch.autograd import Variable
from torch.distributions.normal import Normal
from torch.distributions.uniform import Uniform
from torch.distributions.categorical import Categorical
from torch.distributions.dirichlet import Dirichlet
from torch.distributions.multivariate_normal import MultivariateNormal as MVN
from torch.distributions.kl import kl_divergence
from torch import nn


class FeedForwardNet(nn.Module):
    def __init__(self, **kwargs):
        self.__dict__.update(**kwargs)
        self.layer_shapes = list(zip(self.architecture[:-1], self.architecture[1:]))
        self.number_weights = sum((m + 1) * n for m, n in self.layer_shapes)
        self.weight_dist = Normal(0, self.sigma_w)
        self.noise_dist = Normal(0,self.sigma_noise)
        self.net = None

        self.build_layers()

    def build_layers(self):
        layer_list = []
        layers = self.architecture
        for l, (layer1, layer2) in enumerate(zip(self.architecture[:-1], self.architecture[1:])):

            # if last layer, then exclude bias
            if l == len(layers) - 2:
                use_bias = False
            else:
                use_bias = True

            # Build layers, initialize weights and biases.
            layer_list.append(nn.Linear(layer1, layer2, bias=use_bias))
            layer_list[-1].weight.data.normal_(0, self.sigma_w)
            if use_bias:
                layer_list[-1].bias.data.normal_(0, self.sigma_w)

    def net_to_particle(self):
        return

    def weights_from_particle(self, particle):
        pass

    def standardize_data(self, y):
        return

    def destandardize_data(self, y):
        return

    def normalize_data(self, x):
        return

    def forward(self, x):
        return



