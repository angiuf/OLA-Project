from Source.UCBLearner2 import *
from Source.SplittingLearner import *
from Source.Auxiliary import *
from tqdm import trange


def main():
    env1, model = generate_environment()
    real_conv_rates = model["real_conversion_rates"]
    prices = model["prices"]

    T = 60
    n_exp = 2
    daily_user = 200

    optimal_arm = optimization_algorithm(model, False)  # pull the optimal arm
    print("Optimal_arm: ", optimal_arm)

    optimal_act_rate = mc_simulation(model, real_conv_rates[range(5), optimal_arm], 5, 10000)

    optimal_reward = return_reward(model, prices[range(5), optimal_arm], real_conv_rates[range(5), optimal_arm],
                                   optimal_act_rate, model['real_alpha_ratio'], model['real_quantity'])
    print("Optimal reward: ", optimal_reward)

    learner = UCBLearner2(model.copy())
    split_learner = SplittingLearner()
    instant_regret_obs = [[] for _ in range(n_exp)]
    instant_reward_obs = [[] for _ in range(n_exp)]

    for i in range(n_exp):
        print("Experiment number", i + 1)
        alldata = []

        for t in trange(14):
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


        learner.feat = [[0, 0], [0, 1], [1, 0], [1, 1]]
        split, learners = split_learner.first_split(model.copy(), alldata.copy())
        for class_f in split:
            print(class_f)
        if learners == []:
            learners = [learner]

        for t in trange(T - 14):
            pulled_arms = []
            for ii in range(len(learners)):
                pulled_arms.append(learners[ii].act())
            pulled_arm = []
            for features in [[0, 0], [0, 1], [1, 0], [1, 1]]:
                for jj in range(len(learners)):
                    if features in learners[jj].feat:
                        pulled_arm.append(pulled_arms[jj])

            alpha_ratio = env1.alpha_ratio_otd()
            data = env1.round_single_day_split(daily_user, alpha_ratio, pulled_arm,
                                               [[0, 0], [0, 1], [1, 0], [1, 1]])

            alldata.append(data)

            # run base learner
            pulled_arm_base = learner.act()
            alldata.append(data)
            cr_data = conv_data(data)
            ar_data = alpha_data(data)
            q_data = quantity_data(data)
            learner.update(pulled_arm_base, cr_data, ar_data, q_data)

            for cust in data:
                for ler in learners:
                    if cust[3] in ler.feat:
                        cr_data = conv_data([cust])
                        ar_data = alpha_data([cust])
                        q_data = quantity_data([cust])
                        ler.update(cust[7], cr_data, ar_data, q_data)

            obs_reward = 0
            if len(data):
                for i_ in range(len(data)):
                    obs_reward += np.sum(data[i_][0])

                obs_reward /= len(data)

            instant_regret_obs[i].append(optimal_reward - obs_reward)
            instant_reward_obs[i].append(obs_reward)

            if t % 14 == 0 and t > 0:
                split, learners = split_learner.first_split(model.copy(), alldata.copy())
                for class_f in split:
                    print(class_f)
                if learners == []:
                    learners = [learner]

        learner.reset()

    show_results(instant_regret_obs, "UCB test, second case: regret")
    show_reward(instant_reward_obs, "UCB test, second case: reward")


main()
