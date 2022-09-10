from Source.NonStationaryEnvironment import *
import matplotlib.pyplot as plt


def generate_environment():
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

    model = {"n_prod": 5,
             "n_price": 4,
             "prices": prices,
             "cost": costs,
             "real_alphas": alphas_par,
             "real_alpha_ratio": alphas_par / np.sum(alphas_par),
             "real_conversion_rates": real_conv_rates,
             "real_quantity": 3,
             "secondary_products": secondary_products,
             "real_P": P[:, :, 0] * class_probability[0] + P[:, :, 1] * class_probability[1] + P[:, :, 2] *
                       class_probability[2],
             "lambda_secondary": 0.5,
             "daily_user": 1000
             }
    return env1, model, class_probability


def generate_environment_non_stat():
    average = np.array([[[7, 9, 10], [4, 4, 3.5], [4, 4, 5], [4, 3.5, 5], [1.5, 2, 2]],
                        [[6, 8, 9], [1.5, 2, 2], [3, 3, 4], [2, 2.5, 2.5], [1, 1.5, 1]],
                        [[8, 11, 11.5], [2, 3, 3], [5, 5.5, 6.5], [3, 3, 3.5], [2, 2, 2]]])

    variance = np.array([[[1, 1, 1], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]],
                         [[1, 1, 1], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]],
                         [[1, 1, 1], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]]])

    prices = generate_prices(np.array([7, 2, 4, 3, 1]))

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
    horizon = 199

    env2 = NonStationaryEnvironment(average, variance, prices, costs, lambdas, alphas_par, P, secondary_products,
                                    lambda_secondary=0.5, horizon=horizon)

    real_conv_rates = np.zeros((3, 5, 4))

    for i in range(3):
        for j in range(3):
            real_conv_rates[i] += env2.get_real_conversion_rates(j, i) * class_probability[j]

    model = {"n_prod": 5,
             "n_price": 4,
             "n_phase": average.shape[0],
             "prices": prices,
             "cost": costs,
             "real_alphas": alphas_par,
             "real_alpha_ratio": alphas_par / np.sum(alphas_par),
             "real_conversion_rates": real_conv_rates,
             "real_quantity": 3,
             "secondary_products": secondary_products,
             "real_P": P[:, :, 0] * class_probability[0] + P[:, :, 1] * class_probability[1] + P[:, :, 2] *
                       class_probability[2],
             "lambda_secondary": 0.5,
             "daily_user": 1000
             }
    return env2, model, class_probability, horizon


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


# plots the regret with bands
def show_results(instant_regret_obs, title=""):
    cumulative_regret_obs = np.zeros(len(instant_regret_obs[0]))
    cumulative_regret_obs_std = np.zeros(len(instant_regret_obs[0]))
    instant_regret_obs_new = [[] for _ in range(len(instant_regret_obs))]

    for i in range(len(instant_regret_obs)):
        instant_regret_obs_new[i] = np.cumsum(instant_regret_obs[i])

    instant_regret_obs_new = np.array(instant_regret_obs_new)

    for j in range(len(instant_regret_obs[0])):
        cumulative_regret_obs[j] = np.mean(instant_regret_obs_new[:, j])
        cumulative_regret_obs_std[j] = np.std(instant_regret_obs_new[:, j]) / np.sqrt(len(instant_regret_obs_new[:, j]))

    for i in range(len(instant_regret_obs_new[:, 0])):
        plt.plot(instant_regret_obs_new[i, :], color='C3', alpha=0.1)

    plt.plot(cumulative_regret_obs, color='C3', label='Observed')
    plt.fill_between(range(len(cumulative_regret_obs)), cumulative_regret_obs - 2*cumulative_regret_obs_std,
                     cumulative_regret_obs + 2*cumulative_regret_obs_std, alpha=0.2)
    plt.title(title)
    plt.legend()
    plt.show()


def reward_per_prod(data):
    result = [0 for _ in range(5)]
    number = [0 for _ in range(5)]
    for i in range(len(data)):
        for j in range(5):
            result[j] += data[i][5][j]
            number[j] += data[i][4][j]

    for j in range(5):
        if number[j] > 0:
            result[j] /= len(data)

    return result, number