from Source.Learner import Learner
from Source.UCBLearner2 import UCBLearner2
from Source.GreedyAlgorithm import *
from Source.Auxiliary import *
import numpy as np

class SplittingLearner:
    def __init__(self, model):
        self.model = model

    def split(self, p, lb):  # p are the class probabilities, lb are the lower bounds for aggregated and the split
        return (p[0]*lb[1] + p[1]*lb[2]) - lb[0]

    def evaluate_split(self, data):
        lb = []
        tot_reward = 0
        tot_reward_f_00 = 0
        tot_reward_f_01 = 0
        tot_reward_f_10 = 0
        tot_reward_f_11 = 0
        n_f_00 = 0
        n_f_01 = 0
        n_f_10 = 0
        n_f_11 = 0
        for datum in data:
            tot_reward += np.sum(datum[0])
            if datum[3] == [0,0] or [0,1]:
                tot_reward_f_00 += np.sum(datum[0])
                n_f_00 += 1
            if datum[3] == [1,0] or [1,1]:
                tot_reward_f_01 += np.sum(datum[0])
                n_f_01 += 1
            if datum[3] == [0,0] or [1,0]:
                tot_reward_f_10 += np.sum(datum[0])
                n_f_10 += 1
            if datum[3] == [0,1] or [1,1]:
                tot_reward_f_11 += np.sum(datum[0])
                n_f_11 += 1

        mean_reward = tot_reward/len(data)
        mean_reward_f_00 = tot_reward_f_00 / n_f_00
        mean_reward_f_01 = tot_reward_f_01 / n_f_01
        mean_reward_f_10 = tot_reward_f_10 / n_f_10
        mean_reward_f_11 = tot_reward_f_11 / n_f_11
        lb_tot = mean_reward - np.sqrt(-(np.log(0.9))/(2*len(data)))  # TODO: which lower bound to use?

        lb.append()




