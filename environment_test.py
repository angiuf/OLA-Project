from Environment_Pricing.EnvironmentPricing import EnvironmentPricing
from Environment_Pricing.EnvironmentPricingAggregated import \
    EnvironmentPricingAggregated as EnvironmentPricingAggregated_
from GreedyAlgorithm import *
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

    # Model is a dictionary containing real or estimated parameters:
    # Conversion rates
    # Alpha ratios
    # Mean number of purchased products
    # Secondary product
    # P
    # Lambdas

    model = {
        "alphas": np.random.uniform(0, 1, 6),
        "act_prob": np.random.uniform(0, 1, (5, 5)),
        "conversion_rate": np.random.uniform(0, 1, (5, 4)),
        "quantity": 3,
        "secondary_products": secondary_products,
        "P": np.mean(P, axis=2),
        "lambda_secondary": 0.5
    }

    optimization_algorithm(prices, 5, 4, model)

    env1 = EnvironmentPricingAggregated_(average, variance, prices, costs, lambdas, alphas_par, P, secondary_products,
                                         lambda_secondary=0.5)

    conv_data = env1.round_single_day(100, np.array([0, 0, 0, 0, 0]))


main()
