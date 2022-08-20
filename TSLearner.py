from Learner import Learner
from GreedyAlgorithm import *
import numpy as np


class TSLearner(Learner):
    def __init__(self, n_prod, n_price, prices, model):
        super().__init__(n_prod, n_price, model)
        self.alphas = np.ones((n_prod, n_price))
        self.betas = np.ones((n_prod, n_price))
        self.prices = prices

    def act(self):
        samples = np.array(
            [[np.random.beta(a=self.alphas[i, j], b=self.betas[i, j]) for j in range(self.n_price)] for i in
             range(self.n_prod)])
        self.model['conv_means'] = samples
        arm_pulled = optimization_algorithm(self.prices, self.n_prod, self.n_price, self.model, False, "conv_means")
        return arm_pulled  # act optimsistically towards the sampling

    def update(self, arm_pulled, conv_data):
        super().update(arm_pulled, conv_data)
        for i in range(self.n_prod):
            self.alphas[i, arm_pulled[i]] += np.sum(conv_data[i])
            self.betas[i, arm_pulled[i]] += (len(conv_data[i]) - np.sum(conv_data[i]))
