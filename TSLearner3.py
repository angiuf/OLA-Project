from Learner import Learner
from GreedyAlgorithm import *
import numpy as np


class TSLearner3(Learner):
    def __init__(self, model):
        super().__init__(model)
        self.alphas = np.ones((self.n_prod, self.n_price))  # alphas of the TS
        self.betas = np.ones((self.n_prod, self.n_price))  # betas of the TS
        self.prices = model["prices"]  # prices

        self.clicks_means = np.zeros((5,5))  # means of the alpha ratio for each product
        self.model['clicks_means'] = self.clicks_means


    # Returns the chosen arm
    def act(self):
        samples = np.array(
            [[np.random.beta(a=self.alphas[i, j], b=self.betas[i, j]) for j in range(self.n_price)] for i in
             range(self.n_prod)])
        self.model['cr_means'] = samples
        arm_pulled = optimization_algorithm(self.model, False, rates="cr_means", clicks='clicks_means')
        return arm_pulled  # act optimistically towards the sampling

    # Updates alphas and betas
    def update(self, arm_pulled, conv_data, clicks_data):
        super().update3(arm_pulled, conv_data, clicks_data)
        for i in range(self.n_prod):
            self.alphas[i, arm_pulled[i]] += np.sum(conv_data[i])
            self.betas[i, arm_pulled[i]] += (len(conv_data[i]) - np.sum(conv_data[i]))

        for i in range(self.n_prod):
            data = np.array(clicks_data)
            sum = np.sum(data[i, :])
            for j in range(self.n_prod):
                if sum == 0:
                    self.clicks_means[i, j] = 0.0
                else:
                    self.clicks_means[i, j] = data[i, j]/sum

        self.model['clicks_means'] = self.clicks_means