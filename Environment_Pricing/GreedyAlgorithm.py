import numpy as np

class GreedyAlgorithm():
    def __init__(self, prices):
        self.prices = prices

    def optimization_algorithm(self):
        price_arm = np.zeros(5)
        rewards = np.array([])
        while True:
            prec_rewards = calculate_reward(price_arm)
            for i in range(0,5):
                add_price = np.zeros(5)
                add_price[i] = 1
                rewards[i] = calculate_reward(price_arm + add_price) #Control on index exciding columns
            idx = np.argmax(rewards)
            if rewards[idx] <= prec_reward:
                return price_arm
            else:
                if price_arm[idx] != 3:
                    price_arm[idx] += 1
