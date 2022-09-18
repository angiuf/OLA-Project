from Source.Learner import Learner
from Source.UCBLearner2 import UCBLearner2
from Source.GreedyAlgorithm import *
from Source.Auxiliary import *
from Source.TSLearner2 import TSLearner2
import numpy as np


def split(p0, p1, lb0, lb1, lb_tot):  # p are the class probabilities,
    # lb are the lower bounds for aggregated and the split
    return (p0 * lb0 + p1 * lb1) - lb_tot

def first_split(model, data, ucb_learner, learner_model):
    # The first number indicate the feature, and the second feature indicate its value. So, for example,
    # 00 refers to the case when the first feature is 0.

    model_00 = model.copy()
    model_01 = model.copy()
    model_10 = model.copy()
    model_11 = model.copy()

    if ucb_learner:
        learner_00 = UCBLearner2(model_00)
        learner_01 = UCBLearner2(model_01)
        learner_10 = UCBLearner2(model_10)
        learner_11 = UCBLearner2(model_11)
    else:
        learner_00 = TSLearner2(model_00)
        learner_01 = TSLearner2(model_01)
        learner_10 = TSLearner2(model_10)
        learner_11 = TSLearner2(model_11)

    n = 0
    n_f_00 = 0
    n_f_01 = 0
    n_f_10 = 0
    n_f_11 = 0

    # Split the data
    data_00 = []
    data_01 = []
    data_10 = []
    data_11 = []
    for day in data:
        day_00 = []
        day_01 = []
        day_10 = []
        day_11 = []
        for cust in day:
            n += 1
            if cust[3][0] == 0:
                day_00.append(cust)
                n_f_00 += 1
            else:
                day_01.append(cust)
                n_f_01 += 1

            if cust[3][1] == 0:
                day_10.append(cust)
                n_f_10 += 1
            else:
                day_11.append(cust)
                n_f_11 += 1

        cr_data_00 = conv_data(day_00)
        ar_data_00 = alpha_data(day_00)
        q_data_00 = quantity_data(day_00)
        pulled_arm_00 = day_00[0][7]
        learner_00.update(pulled_arm_00, cr_data_00, ar_data_00, q_data_00)

        cr_data_01 = conv_data(day_01)
        ar_data_01 = alpha_data(day_01)
        q_data_01 = quantity_data(day_01)
        pulled_arm_01 = day_01[0][7]
        learner_01.update(pulled_arm_01, cr_data_01, ar_data_01, q_data_01)

        cr_data_10 = conv_data(day_10)
        ar_data_10 = alpha_data(day_10)
        q_data_10 = quantity_data(day_10)
        pulled_arm_10 = day_10[0][7]
        learner_10.update(pulled_arm_10, cr_data_10, ar_data_10, q_data_10)

        cr_data_11 = conv_data(day_11)
        ar_data_11 = alpha_data(day_11)
        q_data_11 = quantity_data(day_11)
        pulled_arm_11 = day_11[0][7]
        learner_11.update(pulled_arm_11, cr_data_11, ar_data_11, q_data_11)

        data_00.append(day_00)
        data_01.append(day_01)
        data_10.append(day_10)
        data_11.append(day_11)

    # arm_00 = learner_00.act()
    # arm_01 = learner_01.act()
    # arm_10 = learner_10.act()
    # arm_11 = learner_11.act()


    arm_ = optimization_algorithm(learner_model, False, rates="cr_means", alphas="alpha_means",
                                  quantity="quantity_mean")
    arm_00 = optimization_algorithm(model_00, False, rates="cr_means", alphas="alpha_means",
                                    quantity="quantity_mean")
    arm_01 = optimization_algorithm(model_01, False, rates="cr_means", alphas="alpha_means",
                                    quantity="quantity_mean")
    arm_10 = optimization_algorithm(model_10, False, rates="cr_means", alphas="alpha_means",
                                    quantity="quantity_mean")
    arm_11 = optimization_algorithm(model_11, False, rates="cr_means", alphas="alpha_means",
                                    quantity="quantity_mean")

    act_rate_ = mc_simulation(learner_model, [learner_model['cr_means'][i][arm_[i]] for i in range(5)], 5)

    mu_ = return_reward(learner_model, [learner_model['prices'][i][arm_[i]] for i in range(5)],
                        [learner_model['cr_means'][i][arm_[i]] for i in range(5)], act_rate_,
                        learner_model['alpha_means'], learner_model['quantity_mean'])

    act_rate_00 = mc_simulation(model_00, [model_00['cr_means'][i][arm_00[i]] for i in range(5)], 5)

    mu_00 = return_reward(model_00, [model_00['prices'][i][arm_00[i]] for i in range(5)],
                          [model_00['cr_means'][i][arm_00[i]] for i in range(5)], act_rate_00,
                          model_00['alpha_means'], model_00['quantity_mean'])

    act_rate_01 = mc_simulation(model_01, [model_01['cr_means'][i][arm_01[i]] for i in range(5)], 5)

    mu_01 = return_reward(model_01, [model_01['prices'][i][arm_01[i]] for i in range(5)],
                          [model_01['cr_means'][i][arm_01[i]] for i in range(5)], act_rate_01,
                          model_01['alpha_means'], model_01['quantity_mean'])

    act_rate_10 = mc_simulation(model_10, [model_10['cr_means'][i][arm_10[i]] for i in range(5)], 5)

    mu_10 = return_reward(model_10, [model_10['prices'][i][arm_10[i]] for i in range(5)],
                          [model_10['cr_means'][i][arm_10[i]] for i in range(5)], act_rate_10,
                          model_10['alpha_means'], model_10['quantity_mean'])

    act_rate_11 = mc_simulation(model_11, [model_11['cr_means'][i][arm_11[i]] for i in range(5)], 5)

    mu_11 = return_reward(model_11, [model_11['prices'][i][arm_11[i]] for i in range(5)],
                          [model_11['cr_means'][i][arm_11[i]] for i in range(5)], act_rate_11,
                          model_11['alpha_means'], model_11['quantity_mean'])

    lb_tot = hoeff_bound_l(mu_, n)

    lb_00 = hoeff_bound_l(mu_00, n_f_00)
    lb_01 = hoeff_bound_l(mu_01, n_f_01)
    lb_10 = hoeff_bound_l(mu_10, n_f_10)
    lb_11 = hoeff_bound_l(mu_11, n_f_11)

    p_00 = hoeff_bound(n_f_00 / n, n_f_00)
    p_01 = hoeff_bound(n_f_01 / n, n_f_01)
    p_10 = hoeff_bound(n_f_10 / n, n_f_10)
    p_11 = hoeff_bound(n_f_11 / n, n_f_11)

    # Evaluate the splitting condition for the first feature
    split_0 = split(p_00, p_01, lb_00, lb_01, lb_tot)
    # Evaluate the splitting condition for the second feature
    split_1 = split(p_10, p_11, lb_10, lb_11, lb_tot)

    # print(arm_00, arm_01, arm_10, arm_11)
    # print(split_0, split_1)
    # print(mu_00, mu_01, mu_10, mu_11, mu_)
    # print(lb_tot, lb_00, lb_01, lb_10, lb_11)
    # print(p_00, p_01, p_10, p_11)

    if split_0 < 0 and split_1 < 0:
        # print("No splitting")
        return []
    elif split_0 > split_1:
        # print("Splitting 0")
        second_split_00 = second_split(model_00, data_00, 1, ucb_learner, mu_00)
        l00 = []
        if second_split_00[0]:
            # c_00 = [[[0, 0]], [[0, 1]]]
            l00.append(second_split_00[1])
            l00.append(second_split_00[2])
            l00[0].feat = [[0, 0]]
            l00[1].feat = [[0, 1]]
        else:
            # c_00 = [[[0, 0], [0, 1]]]
            l00.append(learner_00)
            l00[0].feat = [[0, 0], [0, 1]]

        second_split_01 =  second_split(model_01, data_01, 1, ucb_learner, mu_01)
        l01 = []
        if second_split_01[0]:
            # c_01 = [[[1, 0]], [[1, 1]]]
            l01.append(second_split_01[1])
            l01.append(second_split_01[2])
            l01[0].feat = [[1, 0]]
            l01[1].feat = [[1, 1]]
        else:
            # c_01 = [[[1, 0], [1, 1]]]
            l01.append(learner_01)
            l01[0].feat = [[1, 0], [1, 1]]

        l_result = []
        l_result.extend(l00)
        l_result.extend(l01)
        return l_result

    else:
        # print("Splitting 1")
        second_split_10 =  second_split(model_10, data_10, 0, ucb_learner, mu_10)
        l10 = []
        if second_split_10[0]:
            # c_10 = [[[0, 0]], [[1, 0]]]
            l10.append(second_split_10[1])
            l10.append(second_split_10[2])
            l10[0].feat = [[0, 0]]
            l10[1].feat = [[1, 0]]
        else:
            # c_10 = [[[0, 0], [1, 0]]]
            l10.append(learner_10)
            l10[0].feat = [[0, 0], [1, 0]]

        second_split_11 =  second_split(model_11, data_11, 0, ucb_learner, mu_11)
        l11 = []
        if second_split_11[0]:
            # c_11 = [[[0, 1]], [[1, 1]]]
            l11.append(second_split_11[1])
            l11.append(second_split_11[2])
            l11[0].feat = [[0, 1]]
            l11[1].feat = [[1, 1]]
        else:
            # c_11 = [[[0, 1], [1, 1]]]
            l11.append(learner_01)
            l11[0].feat = [[0, 1], [1, 1]]

        l_result = []
        l_result.extend(l10)
        l_result.extend(l11)
        return l_result

def second_split(model, data, other_feature, ucb_learner, mu):
    n = 0
    n_f_0 = 0
    n_f_1 = 0

    model_0 = model.copy()
    model_1 = model.copy()
    if ucb_learner:
        learner_0 = UCBLearner2(model_0)
        learner_1 = UCBLearner2(model_1)
    else:
        learner_0 = TSLearner2(model_0)
        learner_1 = TSLearner2(model_1)

    data_0 = []
    data_1 = []
    for day in data:
        day_0 = []
        day_1 = []
        for cust in day:
            n += 1
            if cust[3][other_feature] == 0:
                n_f_0 += 1
                day_0.append(cust)
            if cust[3][other_feature] == 1:
                n_f_1 += 1
                day_1.append(cust)

        cr_data = conv_data(day_0)
        ar_data = alpha_data(day_0)
        q_data = quantity_data(day_0)
        pulled_arm = day_0[0][7]
        learner_0.update(pulled_arm, cr_data, ar_data, q_data)

        cr_data = conv_data(day_1)
        ar_data = alpha_data(day_1)
        q_data = quantity_data(day_1)
        pulled_arm = day_1[0][7]
        learner_1.update(pulled_arm, cr_data, ar_data, q_data)

        data_0.extend(day_0)
        data_1.extend(day_1)

    # arm_0 = learner_0.act()
    # arm_1 = learner_1.act()

    arm_0 = optimization_algorithm(model_0, False, rates="cr_means", alphas="alpha_means",
                                   quantity="quantity_mean")
    arm_1 = optimization_algorithm(model_1, False, rates="cr_means", alphas="alpha_means",
                                   quantity="quantity_mean")

    act_rate_0 = mc_simulation(model_0, [model_0['cr_means'][i][arm_0[i]] for i in range(5)], 5)

    mu_0 = return_reward(model_0, [model_0['prices'][i][arm_0[i]] for i in range(5)],
                         [model_0['cr_means'][i][arm_0[i]] for i in range(5)], act_rate_0,
                         model_0['alpha_means'], model_0['quantity_mean'])

    act_rate_1 = mc_simulation(model_1, [model_1['cr_means'][i][arm_1[i]] for i in range(5)], 5)

    mu_1 = return_reward(model_1, [model_1['prices'][i][arm_1[i]] for i in range(5)],
                         [model_1['cr_means'][i][arm_1[i]] for i in range(5)], act_rate_1,
                         model_1['alpha_means'], model_1['quantity_mean'])

    lb_tot = hoeff_bound_l(mu, n)

    lb_0 = hoeff_bound_l(mu_0, n_f_0)
    lb_1 = hoeff_bound_l(mu_1, n_f_1)

    p_0 = hoeff_bound(n_f_0 / n, n_f_0)
    p_1 = hoeff_bound(n_f_1 / n, n_f_1)

    split_ =  split(p_0, p_1, lb_0, lb_1, lb_tot)

    if split_ > 0:
        return [True, learner_0, learner_1]
    else:
        return [False]


def hoeff_bound(mean, n_z, conf=0.05):
    return mean - np.sqrt(-np.log(conf) / (2 * n_z))


def hoeff_bound_l(mean, n_z, conf=0.05):
    return mean - np.sqrt(-np.log(conf) / (2 * n_z) * 10 ** 2)


def clt_bound(mean, var, n):
    return mean - np.sqrt(var / n) * 1.64
