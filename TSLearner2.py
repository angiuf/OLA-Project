from Learner import Learner
from GreedyAlgorithm import *
import numpy as np


class TSLearner2(Learner):
    def __init__(self, model):
        super().__init__(model)
        self.alphas = np.ones((self.n_prod, self.n_price))  # alphas of the TS
        self.betas = np.ones((self.n_prod, self.n_price))  # betas of the TS
        self.prices = model["prices"]  # prices
        self.alpha_means = np.zeros(self.n_prod+1)  # means of the alpha ratio for each product
        self.quantity_mean = 0  # means of the alpha ratio for each product

        self.model['alpha_means'] = self.alpha_means
        self.model['quantity_mean'] = self.quantity_mean


    # Returns the chosen arm
    def act(self):
        samples = np.array(
            [[np.random.beta(a=self.alphas[i, j], b=self.betas[i, j]) for j in range(self.n_price)] for i in
             range(self.n_prod)])
        self.model['cr_means'] = samples
        arm_pulled = optimization_algorithm(self.model, False, rates="cr_means", alphas='alpha_means', quantity='quantity_mean')
        return arm_pulled  # act optimistically towards the sampling

    # Updates alphas and betas
    def update(self, arm_pulled, conv_data, alpha_data, quantity_data):
        super().update2(arm_pulled, conv_data, alpha_data, quantity_data)
        for i in range(self.n_prod):
            self.alphas[i, arm_pulled[i]] += np.sum(conv_data[i])
            self.betas[i, arm_pulled[i]] += (len(conv_data[i]) - np.sum(conv_data[i]))

        for i in range(self.n_prod+1):
            self.alpha_means[i] = np.mean(self.reward_per_prod_alpha[i])

        self.quantity_mean = np.mean(self.reward_per_quantity)

        self.model['alpha_means'] = self.alpha_means
        self.model['quantity_mean'] = self.quantity_mean

    def printq(self):
        print(self.quantity_mean)

    def printalpha(self):
        print(self.alpha_means)
