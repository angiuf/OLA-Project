import numpy as np
from Environment_Pricing import *


class GreedyAlgorithm:
    def __init__(self, prices, return_reward, n_prices, n_arms):
        self.prices = prices
        self.return_reward = return_reward
        assert (n_prices == len(prices))
        self.n_prices = n_prices
        self.n_arms = n_arms

    def optimization_algorithm(self):
        price_arm = np.zeros(self.n_prices).astype('int')
        rewards = np.zeros(self.n_prices)
        initial_reward = self.return_reward(price_arm)
        prec_reward = initial_reward
        print('Initial reward: ', initial_reward)
        while True:
            max_arms_counter = 0
            for i in range(self.n_prices):
                if price_arm[i] == self.n_arms-1:
                    rewards[i] = 0
                    max_arms_counter += 1
                else:
                    add_price = np.zeros(self.n_prices).astype('int')
                    add_price[i] = 1
                    rewards[i] = self.return_reward(price_arm + add_price)
                    print("Reward of arm: ", price_arm + add_price, "is: ", rewards[i])

            if max_arms_counter == self.n_prices:
                return price_arm

            idx = np.argmax(rewards)

            if rewards[idx] <= prec_reward:
                print('Final arm chosen: ', price_arm)
                return price_arm
            else:
                add_price = np.zeros(self.n_prices).astype('int')
                add_price[idx] = 1
                price_arm = price_arm + add_price
                prec_reward = rewards[idx]
                print('Selected amr: ', price_arm, 'with reward: ', rewards[idx])
