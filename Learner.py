from EnvironmentPricing import *
from EnvironmentPricingAggregated import *
import numpy as np

class Learner:
    def __init__(self, n_prod, n_price, model, n_daily_user):
        self.t = 0
        self.n_price = n_price
        self.n_prod = n_prod
        #self.rewards = []  # list of all rewards
        self.reward_per_prod_price = [[[] for _ in range(n_price)] for _ in range(n_prod)]  # list of list to collect rewards of each single arm
        self.pulled = []
        self.model = model
        self.n_daily_user = n_daily_user

    # we need two function: one that sends actions to the environment, the other that collects the obs and
    # unpdates the inner functioning of the algorithm. Both of this function are specific to the learning algorithm
    # that you're implementing
    def reset(self):  # function to reset everything to 0
        self.__init__(self.n_prod, self.n_price)  # ri-inizializza

    def act(self):
        pass

    def update(self, arm_pulled, conv_data):
        self.t += 1
        #self.rewards.append(reward)
        for i in range(self.n_prod):
            self.reward_per_prod_price[i][arm_pulled[i]].append(conv_data[i])