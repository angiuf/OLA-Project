from Source.Learner import Learner
from Source.GreedyAlgorithm import *
import numpy as np
from Source.CUMSUM import *


class UCBLearner5(Learner):
    def __init__(self, model, alpha=0.01, M=500, eps=0.01, h=[0.5, 0.5, 0.5, 0.5, 0.5]):
        super().__init__(model)
        self.change_detection = [[CUMSUM(M, eps, h[i]) for _ in range(self.n_price)] for i in range(self.n_prod)]
        self.valid_rewards_per_arm = [[[] for _ in range(self.n_price)] for _ in range(self.n_prod)]
        self.alpha = alpha
        self.detections = []
        self.products_detected = []

        # means of the conversion rate for each product at each price
        self.cr_means = np.zeros((self.n_prod, self.n_price))
        # width for each product and each price, initialized at +inf to explore all arms first
        self.conv_widths = np.array([[np.inf for _ in range(self.n_price)] for _ in range(self.n_prod)])
        self.prices = model["prices"]
        self.model['cr_means'] = self.cr_means  # save conversion rates means in the model
        self.model['ucb_cr'] = self.cr_means + self.conv_widths  # save conversion rates means + widths in the model
        self.n_prod_price = np.zeros((self.n_prod, self.n_price))  # times a price has been selected for a product

    def act(self):
        if np.random.binomial(1, 1 - self.alpha):
            arm_pulled = optimization_algorithm(self.model, False, rates="ucb_cr")
            return arm_pulled
        else:
            arm_pulled = np.random.randint(0, self.n_price, self.n_prod)
            return arm_pulled

    # rewards: list of lists of rewards per products
    def update(self, arm_pulled, conv_data, rewards, n_rewards):
        for i in range(self.n_prod):
            if self.change_detection[i][arm_pulled[i]].update(rewards[i], n_rewards[i]):
                self.detections.append(self.t)
                self.products_detected.append(i)
                self.reward_per_prod_price[i][arm_pulled[i]] = []
                self.n_prod_price[i][arm_pulled[i]] = 0
                self.change_detection[i][arm_pulled[i]].reset()

        super().update(arm_pulled, conv_data)

        for i in range(self.n_prod):
            for j in range(self.n_price):
                if len(self.reward_per_prod_price[i][j]):
                    self.cr_means[i, j] = np.mean(self.reward_per_prod_price[i][j])
            self.n_prod_price[i, arm_pulled[i]] += len(conv_data[i])

        for i in range(self.n_prod):
            for j in range(self.n_price):  # update the confidence bound for all arm
                n = self.n_prod_price[i, j]
                N = np.sum(self.n_prod_price[i, :])
                if n > 0:
                    self.conv_widths[i, j] = np.sqrt(2 * np.log(N) / n)
                else:
                    self.conv_widths[i, j] = np.inf

        self.model['ucb_cr'] = self.cr_means + self.conv_widths
        self.model['cr_means'] = self.cr_means

    def print_det(self):
        print(self.detections)
        print(self.products_detected)

    def print(self):
        M = np.zeros((self.n_prod, self.n_price))
        for i in range(self.n_prod):
            for j in range(self.n_price):
                M[i, j] = self.change_detection[i][j].g_minus

        print(M)