from EnvironmentPricing import *
from EnvironmentPricingAggregated import *
import numpy as np

class Learner:
    def __init__(self, n_prod, n_price):
        self.t = 0
        self.n_price = n_price
        self.n_prod = n_prod
        #self.rewards = []  # list of all rewards
        self.pulled = [] #A che serve??? Da Silvia
        #self.model = model

    def rewards(self, to_find):
        self.reward_per_prod_price = [[[[] for _ in range(self.n_price)] for _ in range(self.n_prod)] for _ in range(len(to_find))]  # list of list to collect rewards of each single arm, for each estimate

    # we need two function: one that sends actions to the environment, the other that collects the obs and
    # unpdates the inner functioning of the algorithm. Both of this function are specific to the learning algorithm
    # that you're implementing
    def reset(self):  # function to reset everything to 0
        self.__init__(self.n_prod, self.n_price)  # ri-inizializza

    def act(self):
        pass

    def update(self, arm_pulled, data):
        self.t += 1
        #self.rewards.append(reward)
        print("Number of estimates: ", len(self.reward_per_prod_price))
        for i in range(len(self.reward_per_prod_price)):
            for j in range(self.n_prod):
                self.reward_per_prod_price[i][j][arm_pulled[j]].append(data[i][j]) # Append data for conversion rate for each prod, for each price, for each value you want to estimate