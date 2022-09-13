from Source.UCBLearner2 import *
from Source.SplittingLearner import *
from Source.Auxiliary import *
from tqdm import trange


def main():
    env1, model = generate_environment()
    real_conv_rates = model["real_conversion_rates"]
    prices = model["prices"]

    T = 50
    n_exp = 1
    daily_user = 200

    optimal_arm = optimization_algorithm(model, False)  # pull the optimal arm
    print("Optimal_arm: ", optimal_arm)

    optimal_act_rate = mc_simulation(model, real_conv_rates[range(5), optimal_arm], 5, 10000)

    optimal_reward = return_reward(model, prices[range(5), optimal_arm], real_conv_rates[range(5), optimal_arm],
                                   optimal_act_rate, model['real_alpha_ratio'], model['real_quantity'])
    print("Optimal reward: ", optimal_reward)

    learner = UCBLearner2(model)
    split_learner = SplittingLearner()
    instant_regret_obs = [[] for _ in range(n_exp)]
    instant_reward_obs = [[] for _ in range(n_exp)]

    for i in range(n_exp):
        print("Experiment number", i + 1)
        alldata = []

        for t in trange(T):
            pulled_arm = learner.act()
            alpha_ratio = env1.alpha_ratio_otd()
            data = env1.round_single_day_split(daily_user, alpha_ratio, [pulled_arm for _ in range(4)],
                                               [[0, 0], [0, 1], [1, 0], [1, 1]])
            alldata.append(data)
            cr_data = conv_data(data)
            ar_data = alpha_data(data)
            q_data = quantity_data(data)
            learner.update(pulled_arm, cr_data, ar_data, q_data)

            obs_reward = 0
            if len(data):
                for i_ in range(len(data)):
                    obs_reward += np.sum(data[i_][0])

                obs_reward /= len(data)

            instant_regret_obs[i].append(optimal_reward - obs_reward)
            instant_reward_obs[i].append(obs_reward)

            if t > 0 and t % 14 == 0:
                print(split_learner.first_split(model, alldata))

        learner.reset()

    show_results(instant_regret_obs, "UCB test, second case: regret")
    show_results(instant_reward_obs, "UCB test, second case: reward")


main()
