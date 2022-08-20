from EnvironmentPricing import EnvironmentPricing
from GreedyAlgorithm import *
from EnvironmentPricingAggregated import EnvironmentPricingAggregated
from Learner import *
from UCBLearner import *
from TSLearner import *
import numpy as np
import matplotlib.pyplot as plt


def generate_prices(product_prices):
    prices = np.zeros((len(product_prices), 4))
    changing = np.array([-0.4, -0.2, 0.2, 0.4])
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

    alphas_par = np.array([5, 1, 1, 1, 1, 1])

    np.random.seed(5)
    P = np.random.uniform(0.7, 0.8, size=(5, 5, 3))

    secondary_products = np.array([[1, 4],
                                   [0, 2],
                                   [3, 0],
                                   [2, 4],
                                   [0, 1]])

    env1 = EnvironmentPricing(average, variance, prices, costs, lambdas, alphas_par, P, secondary_products,
                              lambda_secondary=0.5)

    real_conv_rates = np.zeros((5, 4))

    for i in range(3):
        real_conv_rates += env1.get_real_conversion_rates(i) * class_probability[i]

    model = {"alphas": alphas_par,
             "act_prob": np.random.uniform(0, 1, (5, 5)),
             "conversion_rate": real_conv_rates,
             "quantity": 0,
             "secondary_products": secondary_products,
             "P": np.mean(P, axis=2),
             "lambda_secondary": 0.5}

    price_arm=[0,0,0,0,0]
    extr_conversion_rate = model["conversion_rate"][range(5), price_arm]
    MC_simulation(model, extr_conversion_rate, 5)

main()
