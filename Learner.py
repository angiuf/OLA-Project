class Learner:
    def __init__(self, model):
        self.t = 0  # time
        self.n_price = model["n_price"]  # number of prices per product
        self.n_prod = model["n_prod"]  # number of products
        self.reward_per_prod_price = [[[] for _ in range(self.n_price)] for _ in
                                      range(self.n_prod)]  # list of list to collect rewards of each single arm (0, 1)
        self.reward_per_prod_alpha = [[] for _ in range(self.n_prod+1)]  # list of list to collect the first product seen by the users
        self.reward_per_quantity = [] # list to collect the number of objects bought by the users
        self.reward_per_clicks = [[[] for _ in range(self.n_prod)] for _ in range(self.n_prod)]

        self.model_0 = model.copy() #TODO: si deve copiare?
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
            self.reward_per_prod_price[i][arm_pulled[i]].extend(conv_data[i])  # Append data for conversion rate for each prod, for each price

    def update2(self, arm_pulled, conv_data, alpha_data, quantity_data):
        self.t += 1
        for i in range(self.n_prod):
            self.reward_per_prod_price[i][arm_pulled[i]].extend(conv_data[i])  # Append data for conversion rate for each prod, for each price

        for i in range(self.n_prod+1):
            self.reward_per_prod_alpha[i].extend(alpha_data[i])

        self.reward_per_quantity.extend(quantity_data)

    def update3(self, arm_pulled, conv_data, clicks_data):
        self.t += 1
        for i in range(self.n_prod):
            self.reward_per_prod_price[i][arm_pulled[i]].extend(conv_data[i])  # Append data for conversion rate for each prod, for each price

        for i in range(self.n_prod):
            for j in range(self.n_prod):
                self.reward_per_clicks[i][j].extend(clicks_data[i][j])