from UCBLearner4 import *
import numpy as np
import matplotlib.pyplot as plt
import math
from GenerateEnvironment import *


def main():
    env1, model, class_probability, T = generate_environment_non_stat()
    real_conv_rates = model["real_conversion_rates"]
    prices = model["prices"]
    phase_size = T/model["n_phase"]

    n_exp = 1
    daily_user = 50

    optimal_arm = np.zeros((model["n_phase"], model["n_prod"])).astype(int)
    optimal_reward = np.zeros(model["n_phase"])

    for i in range(model["n_phase"]):
        optimal_arm[i] = optimization_algorithm(model, False, phase = i)  # pull the optimal arm
        print("Optimal_arm: ", optimal_arm)

        optimal_act_rate = MC_simulation(model, real_conv_rates[i, range(5), optimal_arm[i]], 5, 10000)

        optimal_reward[i] = return_reward(model, prices[range(5), optimal_arm[i]],
                                       real_conv_rates[i, range(5), optimal_arm[i]], optimal_act_rate, model['real_alpha_ratio'], model['real_quantity'])
        print("Optimal reward: ", optimal_reward)

    learner = UCBLearner4(model, 33)
    instant_regret_rew = [[] for _ in range(n_exp)]
    instant_regret_obs = [[] for _ in range(n_exp)]

    for i in range(n_exp):
        print("Experiment number", i)
        for t in range(4):
            arm = [t, t, t, t, t]
            alpha_ratio = env1.alpha_ratio_otd()
            data = env1.round_single_day(daily_user, alpha_ratio, arm, class_probability)
            env_data = conv_data(data)
            learner.update(arm, env_data)

            obs_reward = 0
            if len(data):
                for i_ in range(len(data)):
                    obs_reward += np.sum(data[i_][0])

                obs_reward /= len(data)

            print("Pulled_arm: ", arm)

            instant_regret_obs[i].append(optimal_reward[0] - obs_reward)
            print("Time: ", t)

        #TODO: controllare perché la fase non funziona
        for t in range(T):
            phase = int(t+4 / phase_size)
            print("Phase: ", phase)
            pulled_arm = learner.act()
            alpha_ratio = env1.alpha_ratio_otd()
            data = env1.round_single_day(daily_user, alpha_ratio, pulled_arm, class_probability)
            env_data = conv_data(data)
            learner.update(pulled_arm, env_data)

            obs_reward = 0
            if len(data):
                for i_ in range(len(data)):
                    obs_reward += np.sum(data[i_][0])

                obs_reward /= len(data)

            print("Pulled_arm: ", pulled_arm)

            instant_regret_obs[i].append(optimal_reward[phase] - obs_reward)
            print("Time: ", t+4)
        learner.reset()

    show_results(instant_regret_rew, instant_regret_obs, "UCB test, fourth case")


main()
