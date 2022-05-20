class Learner:
    def __init__(self, n_arms):
        self.t = 0
        self.n_arms = n_arms
        self.rewards = []  # list of all rewards
        self.reward_per_arm = [[] for _ in range(n_arms)]  # list of list to collect rewards of each single arm
        self.pulled = []

    def reset(self):  # function to reset everything to 0
        self.__init__(self.n_arms)

    def act(self):
        pass

    def update(self, arm_pulled, reward):
        self.t += 1
        self.rewards.append(reward)
        self.reward_per_arm[arm_pulled].append(reward)