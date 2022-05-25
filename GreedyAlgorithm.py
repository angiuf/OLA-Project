import numpy as np
from Environment_Pricing.EnvironmentPricing import *
from environment_test import generate_prices
from Learner import *


class GreedyAlgorithm:
    def __init__(self, prices):
        self.prices = prices

    def optimization_algorithm(self):

        average = np.array([[9, 10, 7],
                            [3, 3, 2],
                            [4, 4, 5],
                            [3, 3.5, 3],
                            [1.5, 2, 2]])

        variance = np.array([[1, 1, 1],
                             [0.5, 0.5, 0.5],
                             [0.5, 0.5, 0.5],
                             [0.5, 0.5, 0.5],
                             [0.5, 0.5, 0.5]])

        prices = generate_prices(np.array([8, 3, 5, 4, 2]))

        costs = np.array([0.8, 0.3, 0.5, 0.4, 0.2])

        class_probability = np.array([0.4, 0.2, 0.4])

        lambdas = np.array([1, 2, 3])

        alphas_par = np.array([2, 1, 1, 1, 1, 1])

        P = np.random.uniform(0, 0.5, size=(5, 5, 3))

        secondary_products = np.array([[1, 4],
                                       [0, 2],
                                       [3, 0],
                                       [2, 4],
                                       [0, 1]])

        env = EnvironmentPricing(average, variance, prices, costs, lambdas, alphas_par, P, secondary_products,
                                  lambda_secondary=0.5)
        learner = Learner()

        price_arm = np.array([0,0,0,0,0])

        #alphas = env.alpha_ratio_otd()
        alphas = np.array([0.4,0.1,0.1,0.1,0.1,0.1])
        lambda_secondary = 0.5

        while True:
            prec_rewards = learner.calculate_total_reward(price_arm, alphas, class_probability, lambdas, prices, secondary_products, lambda_secondary, P, average, variance)
            rewards = np.array([0,0,0,0,0])
            for i in range(0,5):
                add_price = np.array([0,0,0,0,0])
                add_price[i] = 1
                if price_arm[i] + add_price[i] == 4:
                    add_price[i] -= 1
                    rewards[i] = learner.calculate_total_reward(price_arm + add_price, alphas, class_probability, lambdas, prices, secondary_products, lambda_secondary, P, average, variance) #Control on index exciding columns
            idx = np.argmax(rewards)
            if rewards[idx] <= prec_rewards:
                return price_arm
            else:
                if price_arm[idx] != 3:
                    price_arm[idx] += 1


