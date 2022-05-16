import numpy as np
from Environment_Pricing import *


class GreedyAlgorithm:
    def __init__(self, prices, env, MC_days, class_probability):
        self.prices = prices
        self.env = env
        self.MC_days = MC_days
        self.class_probability = class_probability

    def optimization_algorithm(self):
        price_arm = np.array([0, 0, 0, 0, 0])
        rewards = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        initial_reward = 0
        for i in range(self.MC_days):
            initial_reward += self.env.round_single_day(1000, self.env.alpha_ratio_otd(), price_arm,
                                                     self.class_probability)
        initial_reward = initial_reward / self.MC_days
        prec_reward = initial_reward
        print('Initial reward', initial_reward)
        while True:
            for i in range(5):
                add_price = np.zeros(5)
                add_price[i] = 1
                rewards[i] = 0
                for j in range(self.MC_days):
                    rewards[i] = rewards[i] + self.env.round_single_day(1000, self.env.alpha_ratio_otd(), price_arm.astype('int32') + add_price.astype('int32'),
                                                            self.class_probability)
                rewards[i] = rewards[i] / self.MC_days

                print(price_arm.astype('int32') + add_price.astype('int'), ":", rewards[i])
            # Control on index exceeding columns
            idx = np.argmax(rewards)
            if rewards[idx] <= prec_reward:
                return price_arm
            else:
                add_price = np.zeros(5)
                add_price[idx] = 1
                price_arm = price_arm + add_price
                prec_reward = rewards[idx]
