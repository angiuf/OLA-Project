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
    P = np.random.uniform(0.4, 0.8, size=(5, 5, 3))

    secondary_products = np.array([[1, 4],
                                   [0, 2],
                                   [3, 0],
                                   [2, 4],
                                   [0, 1]])

    env1 = EnvironmentPricing(average, variance, prices, costs, lambdas, alphas_par, P, secondary_products,
                              lambda_secondary=0.5)


    real_conv_rates = conv_rate = np.zeros((5, 4))

    for i in range(3):
        real_conv_rates += env1.get_real_conversion_rates(i) * class_probability[i]


    model = {"alphas": alphas_par,
             "act_prob": np.random.uniform(0, 1, (5, 5)),
             "conversion_rate": real_conv_rates,
             "quantity": 0,
             "secondary_products": secondary_products,
             "P": np.mean(P, axis=2),
             "lambda_secondary": 0.5}

    T = 200
    daily_user = 2000

    def q(data_):
        tot = 0
        num = 0
        for i_ in range(len(data_)):
            if data_[i_][2] != -1:
                tot += np.sum(data_[i_][1])
                num += np.sum(np.array(data_[i_][1])>0)
        return tot/num

    alpha_ratio = env1.alpha_ratio_otd()
    data0 = env1.round_single_day(daily_user, alpha_ratio, [0,0,0,0,0], class_probability)
    model["quantity"] = q(data0)
    print("Mean estimate: ", model["quantity"])

    optimal_arm = optimization_algorithm(prices, 5, 4, model)  # pull the optimal arm
    print("Optimal_arm: ", optimal_arm)
    optimal_act_rate = MC_simulation(model, real_conv_rates[range(5), optimal_arm], 5)
    print("Optimal activation rates: ", optimal_act_rate)

    optimal_reward = return_reward(model, prices[range(5), optimal_arm],
                                   real_conv_rates[range(5), optimal_arm], optimal_act_rate)
    print("Optimal reward: ", optimal_reward)

    ucb_learner = UCBLearner(5, 4, prices, model)
    instant_regret = []

    def f(data_):
        result = [[] for _ in range(5)]
        for i_ in range(len(data_)):
            for j_ in range(5):
                if data_[i][1][j_] > 0:
                    result[j_].append(1)
                else:
                    result[j_].append(0)
        return result

    def r(data_):
        result = 0
        for i_ in range(len(data_)):
            result += np.sum(data_[i_][0])
        return result/len(data_)



    for t in range(T):
        pulled_arm = ucb_learner.act()

        alpha_ratio = env1.alpha_ratio_otd()
        data = env1.round_single_day(daily_user, alpha_ratio, pulled_arm, class_probability)
        env_data = f(data)
        ucb_learner.update(pulled_arm, env_data)
        rew = r(data)
        print("Pulled_arm:m ", pulled_arm)
        instant_regret.append(optimal_reward - rew)
        print("Time: ", t)
    cumulative_regret = np.cumsum(instant_regret)

    plt.plot(cumulative_regret)
    plt.show()


main()
