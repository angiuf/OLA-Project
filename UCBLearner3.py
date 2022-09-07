from Learner import Learner
from GreedyAlgorithm import *
import numpy as np


class UCBLearner3(Learner):
    def __init__(self, model):
        super().__init__(model)
        self.clicks_means = np.zeros((5, 5))  # means of the alpha ratio for each product
        self.clicks_widths = np.array([[np.inf for _ in range(self.n_price)] for _ in range(self.n_prod)])  # width for each product and each price, initialized at +inf to explore all arms first

        self.prices = model["prices"]

        self.model['ucb_clicks'] = self.clicks_means + self.clicks_widths  # save conversion rates means + widths in the model
        self.model['clicks_means'] = self.clicks_means
        self.n_prod_price = np.zeros((self.n_prod, self.n_price))  # counts number of times a price has been selected for a product

    def act(self):  # select the arm which has the highest upper confidence bound
        arm_pulled = optimization_algorithm(self.model, False, clicks='ucb_clicks')
        return arm_pulled

    def update(self, arm_pulled, clicks_data):
        super().update3(arm_pulled, clicks_data)

        for i in range(self.n_prod):
            data = np.array(clicks_data)
            sum = np.sum(data[i, :])
            for j in range(self.n_prod):
                if sum == 0:
                    self.clicks_means[i, j] = 0.0
                else:
                    self.clicks_means[i, j] = data[i, j]/sum

        for i in range(self.n_prod):
            for j in range(self.n_price):  # update the confidence bound for all arm
                n = self.n_prod_price[i, j]
                if n > 0:
                    self.clicks_widths[i, j] = np.sqrt(
                        2 * np.log(self.t * self.model['daily_user']) / n)  # TODO: log(t) o log(t) * daily_users?
                else:
                    self.clicks_widths[i, j] = np.inf

        self.model['ucb_clicks'] = self.clicks_means + self.clicks_widths
        self.model['clicks_means'] = self.clicks_means
