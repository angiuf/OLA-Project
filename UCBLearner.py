from Learner import Learner
from GreedyAlgorithm import *
import numpy as np

### versione 1
class UCBLearner(Learner):
    def __init__(self, prices, model, estimates):
        super().__init__(prices.shape[0], prices.shape[1])
        self.prices = prices
        self.n_prod_price = np.zeros((self.n_prod, self.n_price)) # counts number of days i've selected that price for that product
        self.model = model
        self.to_find = np.array(list(model))[estimates] #list of strings that tells what we need to estimate
        super().rewards(self.to_find)
        self.estimate_means = np.zeros((len(self.to_find), self.n_prod, self.n_price))  # now are the means of the conversion rate for each product at each price
        self.estimate_widths = np.array([[[np.inf for _ in range(self.n_price)] for _ in range(self.n_prod)] for _ in range(len(self.to_find))])  # inizializzo a +inf tutti i CI; in questo modo I'll firstly explore each arm (ricorda: scelgo arm con ucb più grande)
        for estimate in self.to_find:
            self.model[estimate] = self.estimate_means[[self.to_find].index(estimate), :, :] + self.estimate_widths[[self.to_find].index(estimate), :, :]

    def act(self):  # select the arm which has the highest upper confidence bound
        arm_pulled = optimization_algorithm(self.prices, self.n_prod, self.n_price, self.model)  # a differenza di prima, ora moltiplico per self.prices
        # i.e. ore scelgo arm (i.e. price) che massimizza [estimate(conversion_rate(p))*price]
        # mentre prima sceglievo arm che massimizza [estimate(conversion rate(p))]
        return arm_pulled

    def update(self, arm_pulled, data): #conv_data must be (n, num_products) where n is the number of estimates
        #reward = reward > 0  # (now the reward is 0 if we don't sell, p if we sell the item: qui lo trasformo in 1 se vendo o 0 se non vendo: perchè ? perchè
        super().update(arm_pulled, data)

        for i in range(len(self.to_find)):
            for j in range(self.n_prod):
                self.estimate_means[i, j, arm_pulled[j]] = np.mean(self.reward_per_prod_price[i][j][arm_pulled[j]])  # update the mean of conversion rate of the arm that we pulled
                self.n_prod_price[j, arm_pulled[j]] += 1
        for i in range(len(self.to_find)):
            for j in range(self.n_prod):
                for k in range(self.n_price):  # update the confidence bound for all arm
                    n = self.n_prod_price[j, k]
                    if n > 0:
                        self.estimate_widths[i, j, k] = np.sqrt(2 * np.log(self.t) / n)
                    else:
                        self.estimate_widths[i, j, k] = np.inf

        for estimate in self.to_find:
            self.model[estimate] = self.estimate_means[[self.to_find].index(estimate), :, :] + \
                                   self.estimate_widths[[self.to_find].index(estimate), :, :]

    def arm_rew(self, arm):
        MC_model = self.model
        for estimate in self.to_find:
            MC_model[estimate] = self.estimate_means[[self.to_find].index(estimate), :, :]

        extr_conv = np.zeros(self.n_prod)
        for i in range(self.n_prod):
            extr_conv[i] = self.estimate_means[[self.to_find].index("conversion_rate"), i, arm[i]]

        act_rate = MC_simulation(self.model, extr_conv, self.n_prod)
        return return_reward(self.model, self.prices[range(5), arm], extr_conv, act_rate)
