import numpy as np
import matplotlib.pyplot as plt
from NonStationaryEnvironment import NonStationaryEnvironment
from GenerateEnvironment import *


def main():
    average = np.array([[[7, 9, 10], [2, 3, 3], [4, 4, 5], [3, 3, 3.5], [1.5, 2, 2]],
                       [[4, 5, 6], [0, 1, 2], [1, 1, 2], [0, 2, 2.5], [0, 0.5, 0.5]],
                       [[10, 11, 12], [5, 6, 7], [7, 8, 9], [6, 7, 8], [5, 6, 7]]])

    variance = np.array([[[1, 1, 1], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]],
                        [[2, 2, 2], [0.5, 1.5, 1.5], [0.5, 0.5, 0.5], [2.5, 2.5, 2.5], [0.5, 1.5, 1.5]],
                        [[3, 3, 3], [0.5, 1.5, 2.5], [2.5, 3.5, 4.5], [1.5, 2.5, 3.5], [0.5, 1.5, 2.5]]])

    prices = generate_prices(np.array([8, 3, 5, 4, 2]))

    costs = np.array([0.8, 0.3, 0.5, 0.4, 0.2])

    class_probability = np.array([0.5, 0.3, 0.2])

    lambdas = np.array([1, 2, 3])

    alphas_par = np.array([5, 1, 1, 1, 1, 1])

    np.random.seed(5)
    P = np.random.uniform(0.1, 0.5, size=(5, 5, 3))

    secondary_products = np.array([[1, 4],
                                   [0, 2],
                                   [3, 0],
                                   [2, 4],
                                   [0, 1]])
    horizon = 99

    env2 = NonStationaryEnvironment(average, variance, prices, costs, lambdas, alphas_par, P, secondary_products, lambda_secondary=0.5, horizon=horizon)

    tot_reward = []
    for t in range(horizon):
        print('Time: ', t)
        alphas_ratio = env2.alpha_ratio_otd()
        data = env2.round_single_day(n_daily_users=100, alpha_ratio=alphas_ratio, arms_pulled=[0, 0, 0, 0, 0],
                                     class_probability=class_probability)
        reward = 0
        for i in range(len(data)):
            for j in range(5):
                reward += data[i][0][j] * data[i][1][j]
        tot_reward.append(reward)

    x = np.linspace(0, horizon, num = horizon)
    plt.plot(x, tot_reward)
    plt.fill_between(x, tot_reward, where=x < 33, facecolor='green', interpolate=True)
    plt.fill_between(x, tot_reward, where= x >= 33, facecolor='red', interpolate=True)
    plt.fill_between(x, tot_reward, where= x >= 66, facecolor='blue', interpolate=True)
    plt.show()


main()
