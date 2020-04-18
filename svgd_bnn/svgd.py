import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from svgd_bnn.bnn import *

import math
import time
import logging


class BNNSVGD(BayesianNeuralNetwork):

    def __init__(self, **kwargs):
        self.__dict__.update(**kwargs)

    def _compute_rbf_h(self):
        pdist = torch.nn.PairwiseDistance(self.n_particles, self.n_particles)
        fkernel = torch.zeros(self.n_particles, self.n_particles)
        for i in range(self.n_particles):
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

    def infer(self, verbose=True):
        """ Perform SVGD and collects samples. """

        n_batches = self.n_batches
        start_time = time.time()
        self.n_particles = self.svgd_n_particles
        self.particles = self.weight_dist.sample(torch.Size([self.n_particles, self.n_weights]))
        optimizer = torch.optim.Adam([self.particles], lr=self.svgd_init_lr)
        for epoch in range(1, self.svgd_epochs + 1):
            optimizer.zero_grad()
            if n_batches:
                batch_indices = torch.arange(epoch % n_batches, self.N_train, n_batches)
                update = self.update_phi(batch_indices)
            else:
                update = self.update_phi()
            self.particles.grad = -update
            optimizer.step()
            if verbose and epoch % 10 == 0:
                # logging.info(f'Epoch {epoch} reached.')
                self.log.info(f'Epoch {epoch} reached.')
        else:
            end_time = time.time()
            self.log.info(f' SVGD ended after {self.svgd_epochs} epochs. Time took: {(end_time - start_time):.0f} seconds.')
            self.log.info(f'SVGD BNN training resulted in a RMSE of {self.train_rmse()}')

        # Convert to numpy for evaluation and plotting.
        # self.save(infer_id)
        self.particles = self.particles.data.numpy()
        self.all_particles.append((self.particles, "gaussian"))

    def save(self):
        """ Save particles into memory. """
        torch.save(self.particles, f"history.pt")


class BNNSVGDRegressor(BNNSVGD, BNNRegressor):
    """ BNN inference using SVGD for regression. """
    def __init__(self, **kwargs):
        """ Instantiates the MLP. """
        self.__dict__.update(**kwargs)
        BNNRegressor.__init__(self)
        self.all_particles = []

    def predict_single(self, particles, domain):
        """ Generate BNN's prediction (forward pass) over the domain for each particle. """
        if domain.shape == (len(domain),):
            domain = np.expand_dims(domain, axis=1)
        return np.apply_along_axis(lambda w: self.forward(torch.Tensor(domain), weights=torch.Tensor(w)).numpy(), 1,
                                   particles).T

    def standardize_data(self, y):
        self.standard_mean = y.mean()
        self.standard_std = y.std()
        return (y-self.standard_mean)/self.standard_std

    def destandardize_data(self, y):
        return y*self.standard_std + self.standard_mean

    def normalize_data(self, x):
        self.norm_factors = x.max(axis=0)
        return x/self.norm_factors

    def to_tensor(self, data):
        return torch.from_numpy(data).float()

    def to_numpy(self, data):
        return data.detach().numpy()

    def fit(self, X, Y):
        self.X_train = self.to_tensor(self.normalize_data(X))
        self.Y_train = self.to_tensor(self.standardize_data(Y))
        self.N_train = self.X_train.shape[0]

        self.infer()

    def predict(self, X):
        results = []
        x = X/self.norm_factors
        for particles, pt in self.all_particles:
            results.append(self.predict_single(particles, x).squeeze())
        results = np.array(results).reshape(len(x), -1)
        return self.destandardize_data(results.mean(axis=1)), 2 * results.std(axis=1)*self.standard_std

    def test_neg_log_likelihood(self):
        """ Compute negative log-likelihood of test set. """
        results = np.apply_along_axis(lambda w: self.forward(self.X_test, weights=torch.Tensor(w)).numpy(), 1,
                                      self.particles)
        means = torch.tensor(np.mean(results, axis=0))
        return -1 * MVN(means, self.sigma_noise * torch.eye(self.ydim)).log_prob(self.Y_test).sum()

    def train_rmse(self):
        """ Compute RMSE of train set. """
        results = np.apply_along_axis(lambda w: self.forward(self.X_train, weights=torch.Tensor(w)).numpy(), 1,
                                      self.particles)
        means = torch.tensor(np.mean(results, axis=0))
        return torch.nn.MSELoss()(means, self.X_train)

    def test_rmse(self):
        """ Compute RMSE of test set. """
        results = np.apply_along_axis(lambda w: self.forward(self.x_test, weights=torch.Tensor(w)).numpy(), 1,
                                      self.particles)
        means = torch.tensor(np.mean(results, axis=0))
        return torch.nn.MSELoss()(means, self.X_test)
