from Environment_Pricing.EnvironmentPricing import *
from GreedyAlgorithm import *


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

    # P = np.random.uniform(0, 0.5, size=(5, 5, 3))
    P = np.array([[[0.0649739, 0.04612955, 0.49336166],
                   [0.47322388, 0.35533235, 0.27036368],
                   [0.48912187, 0.36822062, 0.38640022],
                   [0.17084295, 0.24495391, 0.00268557],
                   [0.33528046, 0.03091, 0.20981468]],

                  [[0.11936554, 0.1127669, 0.26352732],
                   [0.05874907, 0.1251394, 0.37978899],
                   [0.22489803, 0.47772489, 0.13532699],
                   [0.36722438, 0.16307842, 0.30432474],
                   [0.43443812, 0.12255244, 0.15406568]],

                  [[0.41263025, 0.27286197, 0.15898169],
                   [0.00743153, 0.1434097, 0.24964268],
                   [0.26627064, 0.06857715, 0.24076834],
                   [0.0023752, 0.05271317, 0.0253689],
                   [0.05211702, 0.49118378, 0.29339685]],

                  [[0.49943068, 0.28209058, 0.3913052],
                   [0.17861604, 0.121365, 0.06428266],
                   [0.05884311, 0.15066135, 0.0833184],
                   [0.10590399, 0.28239109, 0.3079471],
                   [0.26556456, 0.07166189, 0.06201215]],

                  [[0.28036974, 0.34213211, 0.19750375],
                   [0.44924095, 0.00771818, 0.40672069],
                   [0.24910909, 0.48954917, 0.20135123],
                   [0.15062648, 0.21477713, 0.3362258],
                   [0.37888606, 0.36215814, 0.07371165]]])

    secondary_products = np.array([[1, 4],
                                   [0, 2],
                                   [3, 0],
                                   [2, 4],
                                   [0, 1]])

    env = EnvironmentPricing(average, variance, prices, costs, lambdas, alphas_par, P, secondary_products,
                             lambda_secondary=0.5)

    alphas = [0.5, 0.1, 0.1, 0.1, 0.1, 0.1]
    reward = env.calculate_total_reward(np.array([0, 0, 0, 0, 0]), alphas, class_probability)
    print(reward)
    reward = env.round_single_day(1000, alphas, np.array([0, 0, 0, 0, 0]), class_probability)
    print(reward)

    reward = 0
    for d in range(0, 100):
        alphas_rand = env.alpha_ratio_otd()
        reward += env.round_single_day(1000, alphas_rand, np.array([0, 0, 0, 0, 0]), class_probability)

    reward = reward / 100

    print(reward)
    alg = GreedyAlgorithm(prices)
    best_arm = alg.optimization_algorithm()
    print(best_arm)


main()
