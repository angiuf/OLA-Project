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
    changing = np.array([-0.6, -0.4, -0.2, 0])
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

    costs = np.array([1.6, 0.6, 1, 0.8, 0.4])

    class_probability = np.array([0.4, 0.2, 0.4])

    lambdas = np.array([1, 2, 3])

    alphas_par = np.array([5, 1, 1, 1, 1, 1])

    np.random.seed(6)
    P = np.random.uniform(0.1, 0.5, size=(5, 5, 3))

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
             "ucb_conversion_rate": real_conv_rates,
             "quantity": 3,
             "secondary_products": secondary_products,
             "P": P[:, :, 0] * class_probability[0] + P[:, :, 1] * class_probability[1] + P[:, :, 2] *
                  class_probability[2],
             "lambda_secondary": 0.5}

    T = 100
    daily_user = 500

    optimal_arm = optimization_algorithm(prices, 5, 4, model, False, "conversion_rate")  # pull the optimal arm
    print("Optimal_arm: ", optimal_arm)

    optimal_act_rate = MC_simulation(model, real_conv_rates[range(5), optimal_arm], 5)

    optimal_reward = return_reward(model, prices[range(5), optimal_arm],
                                   real_conv_rates[range(5), optimal_arm], optimal_act_rate)
    print("Optimal reward: ", optimal_reward)

    ts_learner = UCBLearner(5, 4, prices, model)
    instant_regret = []

    def f(data_):
        result = [[] for _ in range(5)]
        for i_ in range(len(data_)):
            for j_ in range(5):
                if data[i_][4][j_]:
                    if data_[i_][1][j_] > 0:
                        result[j_].append(1)
                    else:
                        result[j_].append(0)
        return result

    for i in range(4):
        pulled_arm = [i, i, i, i, i]
        alpha_ratio = env1.alpha_ratio_otd()
        data = env1.round_single_day(daily_user, alpha_ratio, pulled_arm, class_probability)
        env_data = f(data)
        ts_learner.update(pulled_arm, env_data)

    for t in range(T):
        pulled_arm = ts_learner.act()
        alpha_ratio = env1.alpha_ratio_otd()
        data = env1.round_single_day(daily_user, alpha_ratio, pulled_arm, class_probability)
        env_data = f(data)
        ts_learner.update(pulled_arm, env_data)
        act_prob = MC_simulation(model, real_conv_rates[range(5), pulled_arm], 5)
        rew = return_reward(model, prices[range(5), pulled_arm],
                                   real_conv_rates[range(5), pulled_arm], act_prob)
        print("Pulled_arm: ", pulled_arm)
        print(rew)
        instant_regret.append(optimal_reward - rew)
        print("Time: ", t)

    cumulative_regret = np.cumsum(instant_regret)

    plt.plot(cumulative_regret)
    plt.show()


main()