from Source.Learner import Learner
from Source.GreedyAlgorithm import *
import numpy as np


class UCBLearner1(Learner):
    def __init__(self, model):
        super().__init__(model)
        self.cr_means = np.zeros(
            (self.n_prod, self.n_price))  # means of the conversion rate for each product at each price
        self.conv_widths = np.array([[np.inf for _ in range(self.n_price)] for _ in range(
            self.n_prod)])  # width for each product and each price, initialized at +inf to explore all arms first
        self.prices = model["prices"]
        self.model['cr_means'] = self.cr_means  # save conversion rates means in the model
        self.model['ucb_cr'] = self.cr_means + self.conv_widths  # save conversion rates means + widths in the model
        self.n_prod_price = np.zeros(
            (self.n_prod, self.n_price))  # counts number of times a price has been selected for a product

    def act(self):  # select the arm which has the highest upper confidence bound
        arm_pulled = optimization_algorithm(self.model, False, rates="ucb_cr")
        return arm_pulled

    def update(self, arm_pulled, conv_data):
        super().update(arm_pulled, conv_data)
        for i in range(self.n_prod):
            if len(self.reward_per_prod_price[i][arm_pulled[i]]):
                self.cr_means[i, arm_pulled[i]] = np.mean(self.reward_per_prod_price[i][arm_pulled[i]])

            self.n_prod_price[i, arm_pulled[i]] += len(conv_data[i])

        for i in range(self.n_prod):
            for j in range(self.n_price):  # update the confidence bound for all arm
                n = self.n_prod_price[i, j]
                N = np.sum(self.n_prod_price[i, :])
                if n > 0:
                    self.conv_widths[i, j] = np.sqrt(2 * np.log(N) / n)
                else:
                    self.conv_widths[i, j] = np.inf

        self.model['ucb_cr'] = self.cr_means + self.conv_widths
        self.model['cr_means'] = self.cr_means
