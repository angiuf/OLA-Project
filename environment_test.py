from Environment_Pricing.EnvironmentPricing import EnvironmentPricing
import numpy as np


def generate_prices(product_prices):
    prices = np.zeros((len(product_prices), 4))
    changing = np.array([-.2, -.1, .1, .2])
    for i in range(len(product_prices)):
        prices[i, :] = np.ones(len(changing)) * product_prices[i] + np.ones(len(changing)) * product_prices[
            i] * changing
    return prices


def main():
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

    env1 = EnvironmentPricing(average, variance, prices, costs, lambdas, alphas_par, P, secondary_products,
                              lambda_secondary=0.5)

    # Test for one day and 10 customers, the arms pulled are the minimum
    alpha_ratio = env1.alpha_ratio_otd()

    #for i in range(20):
    round, means = env1.round_single_day(1000, alpha_ratio, np.array([0, 0, 0, 0, 0]), class_probability)
    print(means)
    #print("Reward:", round)


main()
