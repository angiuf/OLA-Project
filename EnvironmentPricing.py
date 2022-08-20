import numpy as np
from scipy.stats import norm


class EnvironmentPricing:
    # Initialize the environment with the probabilities of purchasing a product wrt the price selected
    def __init__(self, mean, variance, prices, costs, lambdas, alphas_par, P, secondary_products, lambda_secondary):
        self.prices = prices  # (5, 4), prices arms for each product
        self.costs = costs  # (5, 1), costs of each product

        # We assume reservation_price ~ Normal(mean, variance)
        self.mean = mean  # (5, 3), mean of the reservation price for each product and each class
        self.variance = variance  # (5, 3), variance of the reservation price for each product and each class

        self.lam = lambdas  # (3), the number of items bought ~ 1 + Poisson(lam[class])
        self.alphas_par = alphas_par  # (6), parameters of the Dirichlet previously calculated from the expected values
        self.P = P  # (5, 5, 3) click probability of the secondary product from a primary product for each class
        self.secondary_products = secondary_products  # (5, 2) the two secondary products for each product in order
        self.lambda_secondary = lambda_secondary  # fixed probability to observe the second secondary product

    def round_single_day(self, n_daily_users, alpha_ratio, arms_pulled, class_probability):

        data = []
        for u in range(0, n_daily_users):
            data.append(self.round_single_customer(alpha_ratio, arms_pulled, class_probability))
        return data

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
    def alpha_ratio_otd(self):  # alpha ratio of the day
        return np.random.dirichlet(self.alphas_par)

    # Returns the reward of all the items bought by a single customer
    def round_single_customer(self, alpha_ratio, arms_pulled, class_probability):
        seen_primary = np.full(shape=5, fill_value=False)
        extracted_class = np.random.choice(a=[0, 1, 2], p=class_probability)
        current_product = np.random.choice(a=[-1, 0, 1, 2, 3, 4],
                                           p=alpha_ratio)  # CASE -1: the customer goes to a competitor

        number_objects = [0 for _ in range(5)]
        reward_per_object = [0 for _ in range(5)]

        if current_product == -1:
            return [reward_per_object, number_objects, current_product, extracted_class, seen_primary]

        seen_primary[current_product] = True
        self.round_recursive(seen_primary, current_product, extracted_class, arms_pulled, number_objects,
                             reward_per_object)
        return [reward_per_object, number_objects, current_product, extracted_class, seen_primary]

    # Auxiliary function needed in round_single_customer
    def round_recursive(self, seen_primary, primary, extracted_class, arms_pulled, number_objects,
                        reward_per_object):
        reward_per_object[primary], number_objects[primary] = self.round_single_product(primary, arms_pulled[primary],
                                                                                        extracted_class)
        reward = reward_per_object[primary]

        if reward == 0:
            return

        else:
            secondary_1 = self.secondary_products[primary, 0]
            secondary_2 = self.secondary_products[primary, 1]

            if not seen_primary[secondary_1]:
                buy_first_secondary = np.random.binomial(n=1, p=self.P[
                    primary, secondary_1, extracted_class])  # clicks on the shown product to visualize its page
                if buy_first_secondary:
                    seen_primary[secondary_1] = True
                    self.round_recursive(seen_primary, secondary_1, extracted_class, arms_pulled,
                                         number_objects, reward_per_object)

            if not seen_primary[secondary_2]:
                p_ = self.P[primary, secondary_2, extracted_class] * self.lambda_secondary
                buy_second_secondary = np.random.binomial(n=1,
                                                          p=p_)  # clicks on the shown product to visualize its page
                if buy_second_secondary:
                    seen_primary[secondary_2] = True
                    self.round_recursive(seen_primary, secondary_2, extracted_class, arms_pulled,
                                         number_objects, reward_per_object)

        return

    def get_real_conversion_rates(self, class_):
        conv_rate = np.zeros((5, 4))
        for i in range(0, 5):
            for j in range(0, 4):
                conv_rate[i, j] = 1 - norm.cdf(self.prices[i, j], self.mean[i, class_], np.sqrt(self.variance[i, class_]))

        return conv_rate