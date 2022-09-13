from Source.Learner import Learner
from Source.UCBLearner2 import UCBLearner2
from Source.GreedyAlgorithm import *
from Source.Auxiliary import *
import numpy as np


class SplittingLearner:
    def __init__(self):
        self.a = 0

    def split(self, p0, p1, lb0, lb1, lb_tot):  # p are the class probabilities,
        # lb are the lower bounds for aggregated and the split
        return (p0 * lb0 + p1 * lb1) - lb_tot

    def first_split(self, model, data):

        model_00 = model.copy()
        learner_00 = UCBLearner2(model_00)
        model_01 = model.copy()
        learner_01 = UCBLearner2(model_01)
        model_10 = model.copy()
        learner_10 = UCBLearner2(model_10)
        model_11 = model.copy()
        learner_11 = UCBLearner2(model_11)

        # Split the data
        data_00 = []
        data_01 = []
        for day in data:
            day_00 = []
            day_01 = []
            for datum in day:
                if datum[3][0] == 0:
                    day_00.append(datum)
                else:
                    day_01.append(datum)
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

            data_00.extend(day_00)
            data_01.extend(day_01)


        data_10 = []
        data_11 = []
        for day in data:
            day_10 = []
            day_11 = []
            for datum in day:
                if datum[3][1] == 0:
                    day_10.append(datum)
                else:
                    day_11.append(datum)

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

            data_10.extend(day_10)
            data_11.extend(day_11)

        arm_00 = learner_00.act()
        arm_01 = learner_01.act()
        arm_10 = learner_10.act()
        arm_11 = learner_11.act()

        mu_00 = return_reward(model_00, [model_00['prices'][i][arm_00[i]] for i in range(5)],
                              [model_00['cr_means'][i][arm_00[i]] for i in range(5)], model_00['real_P'],
                              model_00['alpha_means'], model_00['quantity_mean'])

        mu_01 = return_reward(model_01, [model_01['prices'][i][arm_01[i]] for i in range(5)],
                              [model_01['cr_means'][i][arm_01[i]] for i in range(5)], model_01['real_P'],
                              model_01['alpha_means'], model_01['quantity_mean'])

        mu_10 = return_reward(model_10, [model_10['prices'][i][arm_10[i]] for i in range(5)],
                              [model_10['cr_means'][i][arm_10[i]] for i in range(5)], model_10['real_P'],
                              model_10['alpha_means'], model_10['quantity_mean'])

        mu_11 = return_reward(model_11, [model_11['prices'][i][arm_11[i]] for i in range(5)],
                              [model_11['cr_means'][i][arm_11[i]] for i in range(5)], model_11['real_P'],
                              model_11['alpha_means'], model_11['quantity_mean'])

        tot_reward = 0
        n = 0
        n_f_00 = 0
        n_f_01 = 0
        n_f_10 = 0
        n_f_11 = 0
        for day in data:
            for datum in day:
                tot_reward += np.sum(datum[0])
                if datum[3][0] == 0:
                    n_f_00 += 1
                if datum[3][0] == 1:
                    n_f_01 += 1
                if datum[3][1] == 0:
                    n_f_10 += 1
                if datum[3][1] == 1:
                    n_f_11 += 1
                n += 1

        mean_reward = tot_reward / n

        lb_tot = hoeff_bound(mean_reward, n)

        lb_00 = hoeff_bound(mu_00, n_f_00)
        lb_01 = hoeff_bound(mu_01, n_f_01)
        lb_10 = hoeff_bound(mu_10, n_f_10)
        lb_11 = hoeff_bound(mu_11, n_f_11)

        p_00 = hoeff_bound(n_f_00 / n, n_f_00)
        p_01 = hoeff_bound(n_f_01 / n, n_f_01)
        p_10 = hoeff_bound(n_f_10 / n, n_f_10)
        p_11 = hoeff_bound(n_f_11 / n, n_f_11)

        split_0 = self.split(p_00, p_01, lb_00, lb_01, lb_tot)
        split_1 = self.split(p_10, p_11, lb_10, lb_11, lb_tot)

        # print(split_0, split_1)
        # print(mean_reward, mu_00, mu_01, mu_10, mu_11)
        # print(lb_tot, lb_00, lb_01, lb_10, lb_11)
        # print(p_00, p_01, p_10, p_11)

        if split_0 < 0 and split_1 < 0:
            #print("No splitting")
            return [[[0, 0], [0, 1], [1, 0], [1, 1]]], []
        elif split_0 > split_1:
            #print("Splitting 0")
            second_split_00 = self.second_split(model_00, data_00, 1)
            l00 = []
            if second_split_00[0]:
                c_00 = [[[0, 0]], [[0, 1]]]
                l00.append(second_split_00[1])
                l00.append(second_split_00[2])
                l00[0].feat = [[0, 0]]
                l00[1].feat = [[0, 1]]

            else:
                c_00 = [[[0, 0], [0, 1]]]
                l00.append(learner_00)
                l00[0].feat = [[0, 0], [0, 1]]

            second_split_01 = self.second_split(model_01, data_01, 1)
            l01 = []
            if second_split_01[0]:
                c_01 = [[[1, 0]], [[1, 1]]]
                l01.append(second_split_01[1])
                l01.append(second_split_01[2])
                l01[0].feat = [[1, 0]]
                l01[1].feat = [[1, 1]]
            else:
                c_01 = [[[1, 0], [1, 1]]]
                l01.append(learner_01)
                l01[0].feat = [[1, 0], [1, 1]]

            result = []
            l_result = []
            result.extend(c_00)
            result.extend(c_01)
            l_result.extend(l00)
            l_result.extend(l01)
            return result, l_result
        else:
            #print("Splitting 1")
            second_split_10 = self.second_split(model_10, data_10, 0)
            l10 = []
            if second_split_10[0]:
                c_10 = [[[0, 0]], [[1, 0]]]
                l10.append(second_split_10[1])
                l10.append(second_split_10[2])
                l10[0].feat = [[0, 0]]
                l10[1].feat = [[1, 0]]

            else:
                c_10 = [[[0, 0], [1, 0]]]
                l10.append(learner_10)
                l10[0].feat = [[0, 0], [1, 0]]

            second_split_11 = self.second_split(model_11, data_11, 0)
            l11 = []
            if second_split_11[0]:
                c_11 = [[[0, 1]], [[1, 1]]]
                l11.append(second_split_11[1])
                l11.append(second_split_11[2])
                l11[0].feat = [[0, 1]]
                l11[1].feat = [[1, 1]]
            else:
                c_11 = [[[0, 1], [1, 1]]]
                l11.append(learner_01)
                l11[0].feat = [[0, 1], [1, 1]]

            result = []
            result.extend(c_10)
            result.extend(c_11)
            l_result = []
            l_result.extend(l10)
            l_result.extend(l11)
            return result, l_result

    def second_split(self, model, data, other_feature):
        tot_reward = 0
        n = 0
        n_f_0 = 0
        n_f_1 = 0
        for datum in data:
            n += 1
            tot_reward += np.sum(datum[0])
            if datum[3][other_feature] == 0:
                n_f_0 += 1
            if datum[3][other_feature] == 1:
                n_f_1 += 1

        mean_reward = tot_reward / n

        data_0 = []
        data_1 = []
        for datum in data:
            if datum[3][0] == 0:
                data_0.append(datum)
            else:
                data_1.append(datum)

        model_0 = model.copy()
        model_1 = model.copy()
        learner_0 = UCBLearner2(model_0)
        learner_1 = UCBLearner2(model_1)

        for datum in data_0:
            cr_data = conv_data([datum])
            ar_data = alpha_data([datum])
            q_data = quantity_data([datum])
            pulled_arm = datum[7]
            learner_0.update(pulled_arm, cr_data, ar_data, q_data)

        for datum in data_1:
            cr_data = conv_data([datum])
            ar_data = alpha_data([datum])
            q_data = quantity_data([datum])
            pulled_arm = datum[7]
            learner_1.update(pulled_arm, cr_data, ar_data, q_data)

        arm_0 = learner_0.act()
        arm_1 = learner_1.act()

        mu_0 = return_reward(model_0, [model_0['prices'][i][arm_0[i]] for i in range(5)],
                             [model_0['cr_means'][i][arm_0[i]] for i in range(5)], model_0['real_P'],
                             model_0['alpha_means'], model_0['quantity_mean'])

        mu_1 = return_reward(model_1, [model_1['prices'][i][arm_1[i]] for i in range(5)],
                             [model_1['cr_means'][i][arm_1[i]] for i in range(5)], model_1['real_P'],
                             model_1['alpha_means'], model_1['quantity_mean'])

        lb_tot = hoeff_bound(mean_reward, n)

        lb_0 = hoeff_bound(mu_0, n_f_0)
        lb_1 = hoeff_bound(mu_1, n_f_1)

        p_0 = hoeff_bound(n_f_0 / n, n_f_0)
        p_1 = hoeff_bound(n_f_1 / n, n_f_1)

        split_ = self.split(p_0, p_1, lb_0, lb_1, lb_tot)

        if split_ > 0:
            return [True, learner_0, learner_1]
        else:
            return [False]


def hoeff_bound(mean, n_z, conf=0.90):
    return mean - np.sqrt(-np.log(conf) / (2 * n_z))


def clt_bound(mean, var, n):
    return mean - np.sqrt(var / n) * 1.64
