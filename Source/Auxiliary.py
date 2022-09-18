from Source.NonStationaryEnvironment import *
import matplotlib.pyplot as plt


# function that generates a standard environment and returns the aggregated model and class probability
def generate_environment(f_c = True):
    average = np.array([[7, 10, 5],
                        [2.5, 1, 3.5],
                        [3.5, 5, 2],
                        [1.5, 4, 3],
                        [1.5, 1, 2.5]])
    variance = np.array([[1, 1, 1],
                         [0.5, 0.5, 0.5],
                         [0.5, 0.5, 0.5],
                         [0.5, 0.5, 0.5],
                         [0.5, 0.5, 0.5]])
    prices = generate_prices(np.array([6, 2, 3.5, 2.5, 1]))
    costs = np.array([1.6, 0.6, 1, 0.8, 0.4])
    class_probability = np.array([0.4, 0.2, 0.4])
    lambdas = np.array([1, 2, 3])
    alphas_par = np.array([[5, 1, 1, 1, 1, 1],
                          [5, 1, 1, 1, 1, 1],
                          [5, 1, 1, 1, 1, 1]])
    np.random.seed(6)
    P = np.random.uniform(0.1, 0.5, size=(5, 5, 3))

    if not f_c:
        P[0, 1, 0] = 0
        P[2, 3, 0] = 0
        P[3, 2, 0] = 0

        P[0, 4, 1] = 0
        P[2, 0, 1] = 0

        P[1, 2, 2] = 0
        P[3, 4, 2] = 0
        P[4, 1, 2] = 0

    secondary_products = np.array([[1, 4],
                                   [0, 2],
                                   [3, 0],
                                   [2, 4],
                                   [0, 1]])

    env1 = EnvironmentPricing(average, variance, prices, costs, lambdas, alphas_par, P, secondary_products,
                              class_probability,
                              lambda_secondary=0.5)

    real_conv_rates = np.zeros((5, 4))
    real_alpha_par = np.zeros(6)

    for i in range(3):
        real_conv_rates += env1.get_real_conversion_rates(i) * class_probability[i]
        real_alpha_par += alphas_par[i,:] * class_probability[i]

    model = {"n_prod": 5,
             "n_price": 4,
             "prices": prices,
             "cost": costs,
             "real_alphas": real_alpha_par,
             "real_alpha_ratio": real_alpha_par / np.sum(real_alpha_par),
             "real_conversion_rates": real_conv_rates,
             "real_quantity": 3,
             "secondary_products": secondary_products,
             "real_P": P[:, :, 0] * class_probability[0] + P[:, :, 1] * class_probability[1] + P[:, :, 2] *
                       class_probability[2],
             "class_probability": class_probability,
             "lambda_secondary": 0.5,
             "daily_user": 200
             }
    return env1, model


# This function given a time horizon returns the non stationary environment with the aggregated model and class, probability
def generate_environment_non_stat(horizon, f_c = True):
    average = np.array([[[7, 9, 10], [4, 4, 3.5], [4, 4, 5], [4, 3.5, 5], [1.5, 2, 2]],
                        [[6, 8, 9], [1.5, 2, 2], [3, 3, 4], [2, 2.5, 2.5], [1, 1.5, 1]],
                        [[8, 11, 11.5], [2, 3, 3], [5, 5.5, 6.5], [3, 3, 3.5], [2, 2, 2]]])

    variance = np.array([[[1, 1, 1], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]],
                         [[1, 1, 1], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]],
                         [[1, 1, 1], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]]])

    prices = generate_prices(np.array([6, 2, 3.5, 2.5, 1]))

    costs = np.array([0.8, 0.3, 0.5, 0.4, 0.2])

    class_probability = np.array([0.5, 0.3, 0.2])

    lambdas = np.array([1, 2, 3])

    alphas_par = np.array([[5, 1, 1, 1, 1, 1],
                          [5, 1, 1, 1, 1, 1],
                          [5, 1, 1, 1, 1, 1]])

    np.random.seed(5)
    P = np.random.uniform(0.1, 0.5, size=(5, 5, 3))

    if not f_c:
        P[0, 1, 0] = 0
        P[2, 3, 0] = 0
        P[3, 2, 0] = 0

        P[0, 4, 1] = 0
        P[2, 0, 1] = 0

        P[1, 2, 2] = 0
        P[3, 4, 2] = 0
        P[4, 1, 2] = 0

    secondary_products = np.array([[1, 4],
                                   [0, 2],
                                   [3, 0],
                                   [2, 4],
                                   [0, 1]])

    env2 = NonStationaryEnvironment(average, variance, prices, costs, lambdas, alphas_par, P, secondary_products,
                                    class_probability,
                                    lambda_secondary=0.5, horizon=horizon)

    real_conv_rates = np.zeros((3, 5, 4))
    real_alpha_par = np.zeros(6)

    for i in range(3):
        for j in range(3):
            real_conv_rates[i] += env2.get_real_conversion_rates(j, i) * class_probability[j]
            real_alpha_par += alphas_par[i, :] * class_probability[i]

    model = {"n_prod": 5,
             "n_price": 4,
             "n_phase": average.shape[0],
             "prices": prices,
             "cost": costs,
             "real_alphas": real_alpha_par,
             "real_alpha_ratio": real_alpha_par / np.sum(real_alpha_par),
             "real_conversion_rates": real_conv_rates,
             "real_quantity": 3,
             "secondary_products": secondary_products,
             "real_P": P[:, :, 0] * class_probability[0] + P[:, :, 1] * class_probability[1] + P[:, :, 2] *
                       class_probability[2],
             "lambda_secondary": 0.5,
             "daily_user": 1000
             }
    return env2, model

def generate_environment_class(c):
    average = np.array([[7, 10, 10],
                        [2.5, 3, 3.5],
                        [3.5, 5, 5],
                        [2, 4, 3],
                        [1.5, 2.5, 2.5]])
    variance = np.array([[1, 1, 1],
                         [0.5, 0.5, 0.5],
                         [0.5, 0.5, 0.5],
                         [0.5, 0.5, 0.5],
                         [0.5, 0.5, 0.5]])
    prices = generate_prices(np.array([6, 2, 3.5, 2.5, 1]))
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
                              class_probability,
                              lambda_secondary=0.5)

    real_conv_rates = np.zeros((5, 4))

    real_conv_rates += env1.get_real_conversion_rates(c)

    model = {"n_prod": 5,
             "n_price": 4,
             "prices": prices,
             "cost": costs,
             "real_alphas": alphas_par,
             "real_alpha_ratio": alphas_par / np.sum(alphas_par),
             "real_conversion_rates": real_conv_rates,
             "real_quantity": lambdas[c] + 1,
             "secondary_products": secondary_products,
             "real_P": P[:, :, c],
             "class_probability": class_probability,
             "lambda_secondary": 0.5,
             "daily_user": 1000
             }
    return env1, model


# given the average prices this function  returns a matrix with all the possible prices for each product
def generate_prices(product_prices):
    prices = np.zeros((len(product_prices), 4))
    changing = np.array([-0.6, -0.2, 0.2, 0.6])
    for i in range(len(product_prices)):
        prices[i, :] = np.ones(len(changing)) * product_prices[i] + np.ones(len(changing)) * product_prices[
            i] * changing
    return prices


# Function that produces 0 1 from the data of the simulation of a day
def conv_data(data_):
    result = [[] for _ in range(5)]
    for i_ in range(len(data_)):
        for j_ in range(5):
            if data_[i_][4][j_]:
                if data_[i_][1][j_] > 0:
                    result[j_].append(1)
                else:
                    result[j_].append(0)
    return result


# Function that returns the data needed to estimate the alpha ratio values
def alpha_data(data_):
    result = [[] for _ in range(6)]
    for i_ in range(len(data_)):
        for j_ in range(6):
            if data_[i_][2] == j_ - 1:
                result[j_].append(1)
            else:
                result[j_].append(0)
    return result


# Function that extract the values needed to estimate the expected mean value
def quantity_data(data_):
    result = []
    for i_ in range(len(data_)):
        for j_ in range(5):
            if data_[i_][5][j_]:
                result.append(data_[i_][1][j_])
    return result


# Function that returns the data needed to estimate the click graph
def clicks_data(data_):
    result = [[[] for _ in range(5)] for _ in range(5)]

    for i in range(len(data_)):
        for j in range(5):
            for k in range(5):
                result[j][k].extend(data_[i][6][j][k])

    return result


def reward_per_prod(data):
    result = [[] for _ in range(5)]
    for i in range(len(data)):
        for j in range(5):
            if data[i][4][j]:
                result[j].append(data[i][5][j])

    return result


# plots the regret with bands
def show_results(collected_data, title, position):
    cumulative_collected_data = np.zeros(len(collected_data[0]))
    cumulative_collected_data_std = np.zeros(len(collected_data[0]))
    collected_data_new = [[] for _ in range(len(collected_data))]

    for i in range(len(collected_data)):
        collected_data_new[i] = np.cumsum(collected_data[i])

    collected_data_new = np.array(collected_data_new)

    for j in range(len(collected_data[0])):
        cumulative_collected_data[j] = np.mean(collected_data_new[:, j])
        cumulative_collected_data_std[j] = np.std(collected_data_new[:, j]) / np.sqrt(len(collected_data_new[:, j]))

    # for i in range(len(collected_data_new[:, 0])):
    #     a = list(collected_data_new[i, :])
    #     a.insert(0,0)
    #     plt.plot(a, color='C3', alpha=0.1)

    cumulative_collected_data = list(cumulative_collected_data)
    cumulative_collected_data_std = list(cumulative_collected_data_std)
    cumulative_collected_data.insert(0, 0)
    cumulative_collected_data_std.insert(0, 0)
    cumulative_collected_data = np.array(cumulative_collected_data)
    cumulative_collected_data_std = np.array(cumulative_collected_data_std)

    p = plt.subplot(position)

    p.plot(cumulative_collected_data, color='C3', label='Observed')
    p.fill_between(range(len(cumulative_collected_data)),
                     cumulative_collected_data - 2*cumulative_collected_data_std,
                     cumulative_collected_data + 2*cumulative_collected_data_std, alpha=0.2)
    p.set_title(title)
    p.legend()


def show_reward(instant_reward_obs, title, position):
    n_exp = len(instant_reward_obs)
    instant_reward_obs_t = []
    for i in range(len(instant_reward_obs[0])):
        instant_reward_obs_t.append([])
        for j in range(n_exp):
            instant_reward_obs_t[i].append(instant_reward_obs[j][i])

    mean = [np.mean(i) for i in instant_reward_obs_t]
    var = [np.std(i) for i in instant_reward_obs_t]
    mean.insert(0, 0)
    var.insert(0, 0)
    mean = np.array(mean)
    var = np.array(var)

    p = plt.subplot(position)

    p.plot(mean, color='C3', label='Observed')
    # plt.fill_between(range(len(mean)),
    #                  mean - var,
    #                  mean + var, alpha=0.2)
    p.set_title(title)

def calculate_reward(data):
    obs_reward = 0
    if len(data):
        for i_ in range(len(data)):
            obs_reward += np.sum(data[i_][0])

        obs_reward /= len(data)

    return obs_reward
