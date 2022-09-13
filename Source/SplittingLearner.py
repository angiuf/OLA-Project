from Source.Learner import Learner
from Source.UCBLearner2 import UCBLearner2
from Source.GreedyAlgorithm import *
from Source.Auxiliary import *
import numpy as np


class SplittingLearner:
    def __init__(self, model):
        self.model = model

    def split(self, p0, p1, lb0, lb1, lb_tot):  # p are the class probabilities,
        # lb are the lower bounds for aggregated and the split
        return (p0 * lb0 + p1 * lb1) - lb_tot

    def first_split(self, data):
        tot_reward = 0
        tot_reward_2 = 0
        tot_reward_f_00 = 0
        tot_reward_f_01 = 0
        tot_reward_f_10 = 0
        tot_reward_f_11 = 0
        tot_reward_f_00_2 = 0
        tot_reward_f_01_2 = 0
        tot_reward_f_10_2 = 0
        tot_reward_f_11_2 = 0
        n = len(data)
        n_f_00 = 0
        n_f_01 = 0
        n_f_10 = 0
        n_f_11 = 0
        for datum in data:
            tot_reward += np.sum(datum[0])
            tot_reward_2 += np.sum(datum[0]) ** 2
            if datum[3][0] == 0:
                tot_reward_f_00 += np.sum(datum[0])
                tot_reward_f_00_2 += np.sum((datum[0])) ** 2
                n_f_00 += 1
            if datum[3][0] == 1:
                tot_reward_f_01 += np.sum(datum[0])
                tot_reward_f_01_2 += np.sum((datum[0])) ** 2
                n_f_01 += 1
            if datum[3][1] == 0:
                tot_reward_f_10 += np.sum(datum[0])
                tot_reward_f_10_2 += np.sum((datum[0])) ** 2
                n_f_10 += 1
            if datum[3][1] == 1:
                tot_reward_f_11 += np.sum(datum[0])
                tot_reward_f_11_2 += np.sum((datum[0])) ** 2
                n_f_11 += 1

        mean_reward = tot_reward / n
        mean_var = (tot_reward_2 - n * mean_reward ** 2) / (n - 1)
        mean_reward_f_00 = tot_reward_f_00 / n_f_00
        mean_reward_f_01 = tot_reward_f_01 / n_f_01
        mean_reward_f_10 = tot_reward_f_10 / n_f_10
        mean_reward_f_11 = tot_reward_f_11 / n_f_11
        var_f_00 = (tot_reward_f_00_2 - n_f_00 * mean_reward_f_00 ** 2) / (n_f_00 - 1)
        var_f_01 = (tot_reward_f_01_2 - n_f_01 * mean_reward_f_01 ** 2) / (n_f_01 - 1)
        var_f_10 = (tot_reward_f_10_2 - n_f_10 * mean_reward_f_10 ** 2) / (n_f_10 - 1)
        var_f_11 = (tot_reward_f_11_2 - n_f_11 * mean_reward_f_11 ** 2) / (n_f_11 - 1)

        lb_tot = hoeff_bound(mean_reward, n)

        lb_00 = hoeff_bound(mean_reward_f_00, n_f_00)
        lb_01 = hoeff_bound(mean_reward_f_01, n_f_01)
        lb_10 = hoeff_bound(mean_reward_f_10, n_f_10)
        lb_11 = hoeff_bound(mean_reward_f_11, n_f_11)

        # lb_tot = clt_bound(mean_reward, mean_var, n)
        #
        # lb_00 = clt_bound(mean_reward_f_00, var_f_00, n_f_00)
        # lb_01 = clt_bound(mean_reward_f_01, var_f_01, n_f_01)
        # lb_10 = clt_bound(mean_reward_f_10, var_f_10, n_f_10)
        # lb_11 = clt_bound(mean_reward_f_11, var_f_11, n_f_11)

        p_00 = hoeff_bound(n_f_00 / n, n_f_00)
        p_01 = hoeff_bound(n_f_01 / n, n_f_01)
        p_10 = hoeff_bound(n_f_10 / n, n_f_10)
        p_11 = hoeff_bound(n_f_11 / n, n_f_11)

        split_0 = self.split(p_00, p_01, lb_00, lb_01, lb_tot)
        split_1 = self.split(p_10, p_11, lb_10, lb_11, lb_tot)

        print(split_0, split_1)
        print(mean_reward_f_00,mean_reward_f_01, mean_reward_f_10, mean_reward_f_11)
        print(lb_00, lb_01, lb_10, lb_11)
        print(p_00, p_01, lb_10, lb_11)

        if split_0 < 0 and split_1 < 0:
            return [[[0, 0], [0, 1], [1, 0], [1, 1]]]
        elif split_0 > split_1:
            data_00 = []
            data_01 = []
            for datum in data:
                if datum[3][0] == 0:
                    data_00.append(datum)
                else:
                    data_01.append(datum)

            if self.second_split(data_00, 1):
                c_00 = [[[0, 0]], [[0, 1]]]
            else:
                c_00 = [[[0, 0], [0, 1]]]

            if self.second_split(data_01, 1):
                c_01 = [[[1, 0]], [[1, 1]]]
            else:
                c_01 = [[[1, 0], [1, 1]]]
            return c_00.extend(c_01)
        else:
            data_10 = []
            data_11 = []
            for datum in data:
                if datum[3][1] == 0:
                    data_10.append(datum)
                else:
                    data_11.append(datum)

            if self.second_split(data_10, 0):
                c_10 = [[[0, 0]], [[1, 0]]]
            else:
                c_10 = [[[0, 0], [1, 0]]]

            if self.second_split(data_11, 0):
                c_11 = [[[0, 1]], [[1, 1]]]
            else:
                c_11 = [[[0, 1], [1, 1]]]
            return c_10.extend(c_11)

    def second_split(self, data, other_feature):
        tot_reward = 0
        tot_reward_2 = 0
        tot_reward_f_0 = 0
        tot_reward_f_1 = 0
        tot_reward_f_0_2 = 0
        tot_reward_f_1_2 = 0
        n = len(data)
        n_f_0 = 0
        n_f_1 = 0
        for datum in data:
            tot_reward += np.sum(datum[0])
            tot_reward_2 += np.sum(datum[0]) ** 2
            if datum[3][other_feature] == 0:
                tot_reward_f_0 += np.sum(datum[0])
                tot_reward_f_0_2 += np.sum((datum[0])) ** 2
                n_f_0 += 1
            if datum[3][other_feature] == 1:
                tot_reward_f_1 += np.sum(datum[0])
                tot_reward_f_1_2 += np.sum((datum[0])) ** 2
                n_f_1 += 1

        mean_reward = tot_reward / n

        mean_reward_f_0 = tot_reward_f_0 / n_f_0
        mean_reward_f_1 = tot_reward_f_1 / n_f_1

        lb_tot = hoeff_bound(mean_reward, n)

        lb_0 = hoeff_bound(mean_reward_f_0, n_f_0)
        lb_1 = hoeff_bound(mean_reward_f_1, n_f_1)

        p_0 = hoeff_bound(n_f_0 / n, n_f_0)
        p_1 = hoeff_bound(n_f_1 / n, n_f_1)

        split_ = self.split(p_0, p_1, lb_0, lb_1, lb_tot)

        if split_ > 0:
            return True
        else:
            return False


def hoeff_bound(mean, n_z, conf=0.05):
    return mean - np.sqrt(-np.log(conf) / (2 * n_z))


def clt_bound(mean, var, n):
    return mean - np.sqrt(var/n) * 1.64
