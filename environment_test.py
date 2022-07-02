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
    P = np.random.uniform(0.1, 0.5, size=(5, 5, 3))

    secondary_products = np.array([[1, 4],
                                   [0, 2],
                                   [3, 0],
                                   [2, 4],
                                   [0, 1]])

    env1 = EnvironmentPricingAggregated(average, variance, prices, costs, lambdas, alphas_par, P, secondary_products,
                              lambda_secondary=0.5)

    model = {"alphas": np.random.uniform(0, 1, 6),
             "act_prob": np.random.uniform(0, 1, (5, 5)),
             "conversion_rate": env1.get_real_conversion_rates(),
             "quantity": 3,
             "secondary_products": secondary_products,
             "P": np.mean(P, axis=2),
             "lambda_secondary": 0.5}


    T = 100

    optimal_arm = optimization_algorithm(prices, 5, 4, model)    # pull the optimal arm
    optimal_act_rate = MC_simulation(model, env1.get_real_conversion_rates()[range(5), optimal_arm], 5)
    optimal_reward = return_reward(model, prices[range(5), optimal_arm], env1.get_real_conversion_rates()[range(5), optimal_arm], optimal_act_rate)

    ucb_learner = UCBLearner(5, 4, prices, model)
    instant_regret = []

    for t in range(T):
        pulled_arm = ucb_learner.act()
        env_data = env1.round_single_day()
        ucb_learner.update()
        rew = ucb_learner.return_reward(pulled_arm)
        instant_regret.append(optimal_reward-rew)
    cumulative_regret = np.cumsum(instant_regret)

    plt.plot(cumulative_regret)


main()



