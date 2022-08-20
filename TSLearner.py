from Learner import Learner
from GreedyAlgorithm import *
import numpy as np


class TSLearner(Learner):
    def __init__(self, model):
        super().__init__(model)
        self.alphas = np.ones((self.n_prod, self.n_price))  # alphas of the TS
        self.betas = np.ones((self.n_prod, self.n_price))  # betas of the TS
        self.prices = model["prices"]  # prices

        self.a_r_p_s_a = [[[[[[] for _ in range(4)] for _ in range(4)] for _ in range(4)] for _ in range(4)] for _ in
                          range(4)]

    # Returns the chosen arm
    def act(self):
        samples = np.array(
            [[np.random.beta(a=self.alphas[i, j], b=self.betas[i, j]) for j in range(self.n_price)] for i in
             range(self.n_prod)])
        self.model['cr_means'] = samples
        arm_pulled = optimization_algorithm(self.prices, self.n_prod, self.n_price, self.model, False, "cr",
                                            self.a_r_p_s_a)
        return arm_pulled  # act optimistically towards the sampling

    # Updates alphas and betas
    def update(self, arm_pulled, conv_data):
        super().update(arm_pulled, conv_data)
        for i in range(self.n_prod):
            self.alphas[i, arm_pulled[i]] += np.sum(conv_data[i])
            self.betas[i, arm_pulled[i]] += (len(conv_data[i]) - np.sum(conv_data[i]))
