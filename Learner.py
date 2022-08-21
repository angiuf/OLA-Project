class Learner:
    def __init__(self, model):
        self.t = 0  # time
        self.n_price = model["n_price"]  # number of prices per product
        self.n_prod = model["n_prod"]  # number of products
        self.reward_per_prod_price = [[[] for _ in range(self.n_price)] for _ in
                                      range(self.n_prod)]  # list of list to collect rewards of each single arm
        self.pulled = []
        self.model_0 = model
        self.model = model

    # we need two functions: one that sends actions to the environment, the other that collects the obs and
    # updates the inner functioning of the algorithm. Both of this function are specific to the learning algorithm
    # that you're implementing
    def reset(self):  # function to reset everything to 0
        self.__init__(self.model_0)  # reset

    def act(self):
        pass

    def update(self, arm_pulled, conv_data):
        self.t += 1
        for i in range(self.n_prod):
            self.reward_per_prod_price[i][arm_pulled[i]].extend(
                conv_data[i])  # Append data for conversion rate for each prod, for each price
