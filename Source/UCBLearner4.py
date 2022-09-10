from Source.Learner import Learner
from Source.GreedyAlgorithm import *
import numpy as np


class UCBLearner4(Learner):
    def __init__(self, model, window_size):
        super().__init__(model)
        # means of the conversion rate for each product at each price
        self.cr_means = np.zeros((self.n_prod, self.n_price))
        # width for each product and each price, initialized at +inf to explore all arms first
        self.conv_widths = np.array([[np.inf for _ in range(self.n_price)] for _ in range(self.n_prod)])
        self.window_size = window_size
        super().set_window_size(self.window_size)

        self.prices = model["prices"]

        self.model['cr_means'] = self.cr_means  # save conversion rates means in the model
        self.model['ucb_cr'] = self.cr_means + self.conv_widths  # save conversion rates means + widths in the model
        self.n_prod_price = np.zeros((self.n_prod, self.n_price))  # times a price has been selected for a product

    def act(self):  # select the arm which has the highest upper confidence bound
        arm_pulled = optimization_algorithm(self.model, False, rates="ucb_cr")
        return arm_pulled

    def update(self, arm_pulled, conv_data):
        super().update4(arm_pulled, conv_data)

        current_data = [[[] for _ in range(self.n_price)] for _ in range(self.n_prod)]
        current_n = [[0 for _ in range(self.n_price)] for _ in range(self.n_prod)]

        for k in range(self.window_size):
            for i in range(self.n_prod):
                for j in range(self.n_price):
                    current_data[i][j].extend(self.reward_per_prod_price_sw[k][i][j])
                    current_n[i][j] += self.n_per_prod_price_sw[k][i][j]

        for i in range(self.n_prod):
            for j in range(self.n_price):
                if len(current_data[i][j]):  # if empty = 0, else mean
                    self.cr_means[i, j] = np.mean(current_data[i][j])
                    self.n_prod_price[i, j] = current_n[i][j]
                else:
                    self.cr_means[i, j] = 0
                    self.n_prod_price[i, j] = 0
                # update the mean of conversion rate of the arm that we pulled

        for i in range(self.n_prod):
            for j in range(self.n_price):  # update the confidence bound for all arm
                n = self.n_prod_price[i, j]
                # N = np.sum(self.n_prod_price[i, :])
                if n > 0:
                    self.conv_widths[i, j] = np.sqrt(2 * np.log(self.t) / n)
                else:
                    self.conv_widths[i, j] = np.inf

        self.model['ucb_cr'] = self.cr_means + self.conv_widths
        self.model['cr_means'] = self.cr_means

    def reset(self):
        self.__init__(self.model_0, self.window_size)
