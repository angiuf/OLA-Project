import numpy as np
from Learner_OLD import Learner

class UCB(Learner):
    def __init__(self, n_arms, prices, T):
        super().__init__(n_arms)
        self.means = np.zeros(n_arms)  # now are the means of the conversion rate
        self.widths = np.array([np.inf for _ in range(n_arms)])
        self.prices = prices
        self.T = T

    def act(self, alphas, number_of_items_sold):  # select the arm which has the highest upper confidence bound
        tot_alphas = np.sum(alphas)
        for prod in range(5):
            #SCRIVERE OBJECTIVE FUNCTION PER SELEZIONARE BOUND
        return idx

    def update(self, arm_pulled, reward):
        reward = reward > 0  # now the reward is 0 if we don't sell, p if we sell the item (at price p)
        super().update(arm_pulled, reward)
        self.means[arm_pulled] = np.mean(self.reward_per_arm[arm_pulled])  # update the mean of the arm that we pulled
        for idx in range(self.n_arms):  # update the confidence bound for all arm
            n = len(self.reward_per_arm[idx])
            if n > 0:
                self.widths[idx] = np.sqrt(2 * np.log(self.t) / n)
            else:
                self.widths[idx] = np.inf