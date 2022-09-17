import numpy as np
from scipy.stats import norm


class EnvironmentPricing:
    # Initialize the environment with the probabilities of purchasing a product wrt the price selected
    def __init__(self, mean, variance, prices, costs, lambdas, alphas_par, P, secondary_products, class_probability, lambda_secondary):
        self.prices = prices  # (5, 4), prices arms for each product
        self.costs = costs  # (5, 1), costs of each product

        # We assume reservation_price ~ Normal(mean, variance)
        self.mean = mean  # (5, 3), mean of the reservation price for each product and each class
        self.variance = variance  # (5, 3), variance of the reservation price for each product and each class

        self.lam = lambdas  # (3), the number of items bought ~ 1 + Poisson(lam[class])
        self.alphas_par = alphas_par  # (3,6), parameters of the Dirichlet previously calculated from the expected values
        self.P = P  # (5, 5, 3) click probability of the secondary product from a primary product for each class
        self.secondary_products = secondary_products  # (5, 2) the two secondary products for each product in order
        self.lambda_secondary = lambda_secondary  # fixed probability to observe the second secondary product
        self.class_probability = class_probability  # probability for the customer to belong to each class

    # Returns data from the simulation of one day, data is a list of observations for each customer
    def round_single_day(self, n_daily_users, alpha_ratio, arms_pulled):
        data = []
        for u in range(0, n_daily_users):
            extracted_class = np.random.choice(a=[0, 1, 2], p=self.class_probability)
            extracted_alphas = alpha_ratio[extracted_class]
            # [0,0] = children ; [1,0] = adult_male ; [1,1] adult_female
            extracted_features = np.zeros(2)
            if extracted_class == 0:
                extracted_features[0] = 0
                extracted_features[1] = np.random.choice(a=[0, 1], p=[0.5, 0.5])
            elif extracted_class == 1:
                extracted_features[0] = 1
                extracted_features[1] = 0
            else:
                extracted_features[0] = 1
                extracted_features[1] = 1
            data.append(self.round_single_customer(extracted_alphas, arms_pulled, extracted_features, extracted_class))
        return data

    def round_single_day_split(self, n_daily_users, alpha_ratio, arms_pulled, arms_features):
        data = []
        for u in range(0, n_daily_users):
            extracted_class = np.random.choice(a=[0, 1, 2], p=self.class_probability)
            extracted_alphas = alpha_ratio[extracted_class, :]
            # [0,0] = children ; [1,0] = adult_male ; [1,1] adult_female
            extracted_features = [0, 0]
            if extracted_class == 0:
                extracted_features[0] = 0
                extracted_features[1] = np.random.randint(0, 2)
            elif extracted_class == 1:
                extracted_features[0] = 1
                extracted_features[1] = 0
            else:
                extracted_features[0] = 1
                extracted_features[1] = 1
            true_arms_pulled = []
            if extracted_features == arms_features[0]:
                true_arms_pulled = arms_pulled[0]
            elif extracted_features == arms_features[1]:
                true_arms_pulled = arms_pulled[1]
            elif extracted_features == arms_features[2]:
                true_arms_pulled = arms_pulled[2]
            else:
                true_arms_pulled = arms_pulled[3]
            data.append(self.round_single_customer(extracted_alphas, true_arms_pulled, extracted_features, extracted_class))
        return data

    # Returns the data of all the items bought by a single customer, in particular
    # [reward for each object, number off objects, first objects seen, class of the user, seen objects]
    def round_single_customer(self, alpha_ratio, arms_pulled, extracted_features, extracted_class):
        seen_primary = np.full(shape=5, fill_value=False)
        """
        extracted_class = np.random.choice(a=[0, 1, 2], p=self.class_probability)
        # [0,0] = children ; [1,0] = adult_male ; [1,1] adult_female
        extracted_features = np.zeros(2)
        if extracted_class == 0:
            extracted_features[0] = 0
            extracted_features[1] = np.random.choice(a=[0, 1], p=0.5)
        elif extracted_class == 1:
            extracted_features[0] = 1
            extracted_features[1] = 0
        else:
            extracted_features[0] = 1
            extracted_features[1] = 1
        """

        current_product = np.random.choice(a=[-1, 0, 1, 2, 3, 4],
                                           p=alpha_ratio)  # CASE -1: the customer goes to a competitor

        number_objects = [0 for _ in range(5)]
        reward_per_object = [0 for _ in range(5)]
        bought_products = np.full(shape=5, fill_value=False)
        clicks = [[[] for _ in range(5)] for _ in range(5)]

        if current_product == -1:
            return [reward_per_object, number_objects, current_product, extracted_features, seen_primary, bought_products,
                    clicks, arms_pulled]

        seen_primary[current_product] = True
        self.round_recursive(seen_primary, current_product, extracted_class, arms_pulled, number_objects,
                             reward_per_object, bought_products, clicks)
        return [reward_per_object, number_objects, current_product, extracted_features, seen_primary, bought_products,
                clicks, arms_pulled]

    # Auxiliary function needed in round_single_customer
    def round_recursive(self, seen_primary, primary, extracted_class, arms_pulled, number_objects,
                        reward_per_object, bought_products, clicks):
        reward_per_object[primary], number_objects[primary] = self.round_single_product(primary, arms_pulled[primary],
                                                                                        extracted_class)
        reward = reward_per_object[primary]

        if reward == 0:
            bought_products[primary] = False
            return

        else:
            bought_products[primary] = True
            secondary_1 = self.secondary_products[primary, 0]
            secondary_2 = self.secondary_products[primary, 1]

            if not seen_primary[secondary_1]:
                click_first_secondary = np.random.binomial(n=1, p=self.P[
                    primary, secondary_1, extracted_class])  # clicks on the shown product to visualize its page
                if click_first_secondary:
                    clicks[primary][secondary_1].append(1)
                    seen_primary[secondary_1] = True
                    self.round_recursive(seen_primary, secondary_1, extracted_class, arms_pulled,
                                         number_objects, reward_per_object, bought_products, clicks)
                else:
                    clicks[primary][secondary_1].append(0)

            if not seen_primary[secondary_2]:
                p_ = self.P[primary, secondary_2, extracted_class] * self.lambda_secondary
                click_second_secondary = np.random.binomial(n=1,
                                                            p=p_)  # clicks on the shown product to visualize its page
                if click_second_secondary:
                    clicks[primary][secondary_2].append(1 / self.lambda_secondary)
                    seen_primary[secondary_2] = True
                    self.round_recursive(seen_primary, secondary_2, extracted_class, arms_pulled,
                                         number_objects, reward_per_object, bought_products, clicks)
                else:
                    clicks[primary][secondary_2].append(0)

        return

    # Returns the reward and the number of objects of a product bought
    def round_single_product(self, product, arm_pulled, extracted_class):
        mean = self.mean[product, extracted_class]
        var = self.variance[product, extracted_class]

        # if it goes below zero this means that the user doesn't buy the product selected
        reservation_price = np.random.normal(loc=mean, scale=np.sqrt(var))

        if self.prices[product, arm_pulled] <= reservation_price:
            number_objects = np.random.poisson(lam=self.lam[extracted_class]) + 1
            reward = (self.prices[product, arm_pulled] - self.costs[product]) * number_objects
            return [round(reward, 2), number_objects]
        else:
            return [0, 0]

    # Returns the alpha ratio of the day
    def alpha_ratio_otd(self):
        alpha_ratios = np.zeros((3,6))
        for i in range(3):
            alpha_ratios[i] = np.random.dirichlet(self.alphas_par[i])
        return alpha_ratios

    def get_real_conversion_rates(self, class_):
        conv_rate = np.zeros((5, 4))
        for i in range(0, 5):
            for j in range(0, 4):
                conv_rate[i, j] = 1 - norm.cdf(self.prices[i, j], self.mean[i, class_],
                                               np.sqrt(self.variance[i, class_]))

        return conv_rate
