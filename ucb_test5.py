from Source.UCBLearner5 import *
from Source.Auxiliary import *
from tqdm import trange

def main():
    T = 99
    env1, model = generate_environment_non_stat(T)
    real_conv_rates = model["real_conversion_rates"]
    prices = model["prices"]
    phase_size = T / model["n_phase"]

    n_exp = 20
    daily_user = 500

    optimal_arm = np.zeros((model["n_phase"], model["n_prod"])).astype(int)
    optimal_reward = np.zeros(model["n_phase"])

    for i in range(model["n_phase"]):
        optimal_arm[i] = optimization_algorithm(model, False, phase=i)  # pull the optimal arm

        optimal_act_rate = mc_simulation(model, real_conv_rates[i, range(5), optimal_arm[i]], 5, 1000)

        optimal_reward[i] = return_reward(model, prices[range(5), optimal_arm[i]],
                                          real_conv_rates[i, range(5), optimal_arm[i]], optimal_act_rate,
                                          model['real_alpha_ratio'], model['real_quantity'])

    print("Optimal reward: ", optimal_reward)
    print("Optimal_arm: ", optimal_arm)

    learner = UCBLearner5(model)
    instant_regret_obs = [[] for _ in range(n_exp)]
    instant_reward_obs = [[] for _ in range(n_exp)]


    for i in range(n_exp):
        print("Experiment number", i+1)

        for t in trange(T):
            phase = int(t / phase_size)
            pulled_arm = learner.act()
            alpha_ratio = env1.alpha_ratio_otd()
            data = env1.round_single_day(daily_user, alpha_ratio, pulled_arm)
            env_data = conv_data(data)
            rewards = reward_per_prod(data)
            learner.update(pulled_arm, env_data, rewards)

            obs_reward = 0
            if len(data):
                for i_ in range(len(data)):
                    obs_reward += np.sum(data[i_][0])

                obs_reward /= len(data)

            instant_regret_obs[i].append(optimal_reward[phase] - obs_reward)
            instant_reward_obs[i].append(obs_reward)

        learner.reset()
        env1.t = 0

    show_results(instant_regret_obs, "UCB test, fifth case: regret")
    show_reward(instant_reward_obs, "UCB test, fifth case: reward")



main()
