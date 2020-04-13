import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from svgd_bnn.bnn import *

import math
import time

class BNNSVGD(BayesianNeuralNetwork):

    def __init__(self, **kwargs):
        pass

    def _compute_rbf_h(self):
        pdist = torch.nn.PairwiseDistance(self.n_particles, self.n_particles)
        fkernel = torch.zeros(self.n_particles, self.n_particles)
        for i in range(self.particles):
            fkernel[i] = pdist(self.particles[i], self.particles)

        fkernel = fkernel.triu(diagonal=1).flatten()
        med = fkernel[fkernel.nonzero()].median()
        return (med ** 2) / math.log(self.n_particles)

    def kernel_rbf(self, x1, x2, h):
        """ Compute the RBF kernel: k(x, x') = exp(-1/h * l2-norm(x, x')). """
        k = torch.norm(x1 - x2)
        k = (k ** 2) / -h
        return torch.exp(k)

    def update_phi(self, batch_indices=None):
        """ Computes a single SVGD epoch and updates the particles. """
        h = self._compute_rbf_h()
        kernel = torch.zeros(self.n_particles, self.n_particles)

        # Repulsive term
        gradk_matrix = torch.zeros(self.n_particles, self.n_particles, self.n_weights)
        for i in range(self.n_particles):
            grad_each_i = torch.zeros(self.n_particles, self.n_weights)
            for j in range(self.n_particles):
                tempw = Variable(self.particles[j], requires_grad=True)
                tempw.grad = None
                k = self.kernel_rbf(tempw, self.particles[i], h)
                kernel[j, i] = k
                k.backward()
                grad_each_i[j] = tempw.grad
            gradk_matrix[i] = grad_each_i

        # Smoothed gradient term
        logp_matrix = torch.zeros(self.n_particles, self.n_weights)
        for j in range(self.n_particles):
            self.weights = Variable(self.particles[j], requires_grad=True)
            self.weights.grad = None
            self.log_posterior(batch_indices).backward()
            logp_matrix[j] = self.weights.grad
        update = logp_matrix.unsqueeze(dim=0).repeat(self.n_particles, 1, 1)
        for i in range(self.n_particles):
            update[i] *= kernel[:, i].unsqueeze(dim=1)

        update += gradk_matrix
        update = update.mean(dim=1)
        return update

    def save(self, infer_id):
        """ Save particles into memory. """
        torch.save(self.particles, f"history/{self.uid}_svgd{infer_id}.pt")