from Source.UCBLearner2 import *
from Source.Auxiliary import *


def main():
    env1, model, class_probability = generate_environment()
    real_conv_rates = model["real_conversion_rates"]
    prices = model["prices"]

    T = 60
    n_exp = 20
    daily_user = 500

    optimal_arm = optimization_algorithm(model, False)  # pull the optimal arm
    print("Optimal_arm: ", optimal_arm)

    optimal_act_rate = mc_simulation(model, real_conv_rates[range(5), optimal_arm], 5, 10000)

    optimal_reward = return_reward(model, prices[range(5), optimal_arm], real_conv_rates[range(5), optimal_arm],
                                   optimal_act_rate, model['real_alpha_ratio'], model['real_quantity'])
    print("Optimal reward: ", optimal_reward)

    learner = UCBLearner2(model)
    instant_regret_obs = [[] for _ in range(n_exp)]

    for i in range(n_exp):
        print("Experiment number", i)

        for t in range(T):
            pulled_arm = learner.act()
            alpha_ratio = env1.alpha_ratio_otd()
            data = env1.round_single_day(daily_user, alpha_ratio, pulled_arm, class_probability)
            cr_data = conv_data(data)
            ar_data = alpha_data(data)
            q_data = quantity_data(data)
            learner.update(pulled_arm, cr_data, ar_data, q_data)

            obs_reward = 0
            if len(data):
                for i_ in range(len(data)):
                    obs_reward += np.sum(data[i_][0])

                obs_reward /= len(data)

            print("Pulled_arm: ", pulled_arm)

            instant_regret_obs[i].append(optimal_reward - obs_reward)
            print("Time: ", t)
        learner.reset()

    show_results(instant_regret_obs, "UCB test, second case")


main()
