from Source.UCBLearner2 import *
from Source.SplittingLearner import *
from Source.Auxiliary import *
from tqdm import trange


# TODO: implement all the algorithms with not fully connected click probability graph
# TODO: check splitting condition (always splitting)
# TODO: report

def main():
    fully_i_regret, fully_i_reward, opt_rew = run()
    not_fully_i_regret, not_fully_i_reward, not_fully_opt_rew = run(False)

    plt.figure(1, (16, 9))
    plt.suptitle("UCB test, split case")

    show_results(fully_i_regret, "Fully connected: regret", 221)
    show_results(not_fully_i_regret, "Not fully connected: regret", 222)
    show_reward(fully_i_reward, opt_reward=opt_rew, title="Fully connected: reward", position=223)
    show_reward(not_fully_i_reward, opt_reward=not_fully_opt_rew, title="Not fully connected: reward", position=224)

    plt.show()


def run(f_c=True):
    env1, model = generate_environment(f_c)
    real_conv_rates = model["real_conversion_rates"]
    prices = model["prices"]

    T = 180
    n_exp = 20
    daily_user = 200

    all_features = [[0, 0], [0, 1], [1, 0], [1, 1]]

    optimal_reward = 0
    optimal_reward_c = np.zeros(3)

    for c in range(3):
        env_c, model_c = generate_environment_class(c, f_c)
        real_conv_rates_c = model_c["real_conversion_rates"]
        prices_c = model_c["prices"]

        optimal_arm_c = optimization_algorithm(model_c, False)  # pull the optimal arm
        print("Optimal_arm of class ", c, " : ", optimal_arm_c)

        optimal_act_rate_c = mc_simulation(model_c, real_conv_rates_c[range(5), optimal_arm_c], 5, 10000)

        optimal_reward_c[c] = return_reward(model_c, prices_c[range(5), optimal_arm_c],
                                            real_conv_rates_c[range(5), optimal_arm_c],
                                            optimal_act_rate_c, model_c['real_alpha_ratio'], model_c['real_quantity'])

        # optimal_reward_c[c] = simulate_arm_reward(env_c, daily_user, optimal_arm_c)

        print("Optimal reward of class ", c, " : ", optimal_reward_c[c])

        optimal_reward += optimal_reward_c[c] * model["class_probability"][c]

    print("\nOptimal reward: ", optimal_reward)

    learner = UCBLearner2(model.copy())
    instant_regret_obs = [[] for _ in range(n_exp)]
    instant_reward_obs = [[] for _ in range(n_exp)]

    for i in range(n_exp):
        print("Experiment number", i + 1)
        alldata = []

        for t in trange(14):
            pulled_arm = learner.act()
            alpha_ratio = env1.alpha_ratio_otd()
            data = env1.round_single_day_split(daily_user, alpha_ratio, [pulled_arm for _ in range(4)],
                                               all_features)
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

        learner.feat = all_features
        learners = first_split(model.copy(), alldata.copy(), True, learner.model.copy())

        for ler in learners:
            print(ler.feat)
        if learners == []:
            learners = [learner]

        for t in trange(T - 14):
            pulled_arms = []
            for ii in range(len(learners)):
                pulled_arms.append(learners[ii].act())
            pulled_arm = []
            for features in all_features:
                for jj in range(len(learners)):
                    if features in learners[jj].feat:
                        pulled_arm.append(pulled_arms[jj])

            alpha_ratio = env1.alpha_ratio_otd()
            data = env1.round_single_day_split(daily_user, alpha_ratio, pulled_arm,
                                               all_features)

            alldata.append(data)

            # run base learner
            # day_0_0 = customers with feature [0,0] ...
            day_0_0 = []
            day_0_1 = []
            day_1_0 = []
            day_1_1 = []
            for cust in data:
                if cust[3] == [0, 0]:
                    day_0_0.append(cust)
                elif cust[3] == [0, 1]:
                    day_0_1.append(cust)
                elif cust[3] == [1, 0]:
                    day_1_0.append(cust)
                else:
                    day_1_1.append(cust)

            day_tot = {0: day_0_0,
                       1: day_0_1,
                       2: day_1_0,
                       3: day_1_1,
                       }
            for feat in all_features:
                cr_data = conv_data(day_tot[all_features.index(feat)])
                ar_data = alpha_data(day_tot[all_features.index(feat)])
                q_data = quantity_data(day_tot[all_features.index(feat)])
                learner.update(day_tot[all_features.index(feat)][0][7], cr_data, ar_data, q_data)
                for ler in learners:
                    if feat in ler.feat:
                        ler.update(day_tot[all_features.index(feat)][0][7], cr_data, ar_data, q_data)

            obs_reward = 0
            if len(data):
                for i_ in range(len(data)):
                    obs_reward += np.sum(data[i_][0])

                obs_reward /= len(data)

            instant_regret_obs[i].append(optimal_reward - obs_reward)
            instant_reward_obs[i].append(obs_reward)

            if t % 14 == 0 and t > 0:
                learners = first_split(model.copy(), alldata.copy(), True, learner.model.copy())
                for ler in learners:
                    print(ler.feat)
                if learners == []:
                    learners = [learner]

        learner.reset()

    return instant_regret_obs, instant_reward_obs, optimal_reward


main()
