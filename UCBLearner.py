from Learner import Learner
from GreedyAlgorithm import *
import numpy as np

### versione 1
class UCBLearner(Learner):
    def __init__(self, n_prod, n_price, prices, T):
        super().__init__(n_prod, n_price)
        self.conv_means = np.zeros((n_prod, n_price))  # now are the means of the conversion rate for each product at each price
        self.conv_widths = np.array([[np.inf for _ in range(n_price)] for _ in range(n_prod)])  # inizializzo a +inf tutti i CI; in questo modo I'll firstly explore each arm (ricorda: scelgo arm con ucb più grande)
        self.prices = prices
        self.T = T
        self.model['conversion_rate'] = self.conv_means + self.conv_widths
        self.n_prod_price = np.zeros((n_prod, n_price)) # counts number of days i've selected that price for that product

    def act(self):  # select the arm which has the highest upper confidence bound
        arm_pulled = optimization_algorithm(self.prices, self.n_prod, self.n_price, self.model)  # a differenza di prima, ora moltiplico per self.prices
        # i.e. ore scelgo arm (i.e. price) che massimizza [estimate(conversion_rate(p))*price]
        # mentre prima sceglievo arm che massimizza [estimate(conversion rate(p))]
        return arm_pulled

    def update(self, arm_pulled, conv_data):
        #reward = reward > 0  # (now the reward is 0 if we don't sell, p if we sell the item: qui lo trasformo in 1 se vendo o 0 se non vendo: perchè ? perchè

        super().update(arm_pulled, conv_data)
        for i in range(self.n_prod):
            self.conv_means[i, arm_pulled[i]] = np.mean(
                self.reward_per_prod_price[i][arm_pulled[i]])  # update the mean of conversion rate of the arm that we pulled
            self.n_prod_price[i, arm_pulled[i]] += 1

        for i in range(self.n_prod):
            for j in range(self.n_price):  # update the confidence bound for all arm
                n = self.n_prod_price[i,j]
                if n > 0:
                    self.conv_widths[i,j] = np.sqrt(2 * np.log(self.t) / n)
                else:
                    self.conv_widths[i,j] = np.inf