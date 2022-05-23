import numpy as np
from scipy.stats import norm
import copy


class EnvironmentPricing:
    # Initialize the environment with the probabilities of purchasing a product wrt the price selected
    def __init__(self, mean, variance, prices, costs, lambdas, alphas_par, P, secondary_products, lambda_secondary,
                 class_probability):
        self.prices = prices  # (5, 4), prices arms for each product
        self.costs = costs  # (5,1), costs of each product

        # We assume reservation_price ~ Normal(mean, variance)
        self.mean = mean  # (5, 3), mean of the reservation price for each product and each class
        self.variance = variance  # (5, 3), variance of the reservation price for each product and each class

        self.lam = lambdas  # (3), the number of items bought ~ 1 + Poisson(lam[class])
        self.alphas_par = alphas_par  # (6), parameters of the Dirichlet previously calculated from the expected values
        self.P = P  # (5,5,3) click probability of the secondary product from a primary product for each class
        self.secondary_products = secondary_products  # (5,2) the two secondary products for each product in order
        self.lambda_secondary = lambda_secondary  # fixed probability to observe the second secondary product
        self.class_probability = class_probability

    # Returns the reward of a single product bought
    def round_single_product(self, product, arm_pulled, extracted_class):
        mean = self.mean[product, extracted_class]
        var = self.variance[product, extracted_class]

        # if it goes below zero this means that the user doesn't buy the product selected
        reservation_price = np.random.normal(loc=mean, scale=np.sqrt(var))

        if self.prices[product, arm_pulled] <= reservation_price:
            number_objects = np.random.poisson(lam=self.lam[extracted_class]) + 1
            reward = (self.prices[product, arm_pulled] - self.costs[product]) * number_objects
            return round(reward, 2)
        else:
            return 0

    # Returns the reward of all the items bought by a single customer
    def round_single_customer(self, alpha_ratio, arms_pulled, class_probability):
        seen_primary = np.full(shape=len(self.costs), fill_value=False)
        extracted_class = np.random.choice(a=list(range(len(self.lam))), p=class_probability)
        current_product = np.random.choice(a=[-1] + list(range(len(self.costs))),
                                           p=alpha_ratio)  # CASE -1: the customer goes to a competitor

        if current_product == -1:
            return -1  # since the customer didn't visit our site, we don't consider him when learning

        seen_primary[current_product] = True
        return round(self.round_recursive(seen_primary, current_product, extracted_class, arms_pulled), 2)

    # Auxiliary function needed in round_single_customer
    def round_recursive(self, seen_primary, primary, extracted_class, arms_pulled):
        reward_until_now = 0
        reward = self.round_single_product(primary, arms_pulled[primary], extracted_class)

        if reward == 0:
            return reward_until_now

        else:
            reward_until_now += reward
            secondary_1 = self.secondary_products[primary, 0]
            secondary_2 = self.secondary_products[primary, 1]

            s1 = copy.deepcopy(seen_primary)
            s2 = copy.deepcopy(seen_primary)

            if not seen_primary[secondary_1]:
                click_first_secondary = np.random.binomial(n=1, p=self.P[
                    primary, secondary_1, extracted_class])  # clicks on the shown product to visualize its page
                if click_first_secondary:
                    seen_primary[secondary_1] = True
                    reward_until_now += self.round_recursive(s1, secondary_1,
                                                             extracted_class, arms_pulled)

            if not seen_primary[secondary_2]:
                p_ = self.P[primary, secondary_2, extracted_class] * self.lambda_secondary
                click_second_secondary = np.random.binomial(n=1,
                                                            p=p_)  # clicks on the shown product to visualize its page
                if click_second_secondary:
                    seen_primary[secondary_2] = True
                    reward_until_now += self.round_recursive(s2, secondary_2,
                                                             extracted_class, arms_pulled)
            return reward_until_now

    def round_single_day(self, n_daily_users, alpha_ratio, arms_pulled, class_probability):
        daily_reward = 0
        effective_users = 0

        for u in range(0, n_daily_users):
            reward_single_cust = self.round_single_customer(alpha_ratio, arms_pulled, class_probability)
            if reward_single_cust != -1:
                daily_reward += reward_single_cust
                effective_users += 1

        if effective_users == 0:
            return 0.0
        else:
            return daily_reward / effective_users

    # Returns the alpha ratio of the day
    def alpha_ratio_otd(self):  # alpha ratio of the day
        return np.random.dirichlet(self.alphas_par)

    def calculate_reward(self, seen_primary, primary, arms_pulled, user_class):
        if seen_primary[primary]:
            return 0
        else:
            seen_primary[primary] = True
            first_secondary = self.secondary_products[primary, 0]
            second_secondary = self.secondary_products[primary, 1]

            buy_mean = self.mean[primary, user_class]
            buy_var = self.variance[primary, user_class]

            # if arms_pulled[primary] == 4:
            #     arms_pulled[primary] -= 1
            buy_prob = 1 - norm.cdf(self.prices[primary, arms_pulled[primary]], buy_mean, buy_var)

            seen1 = copy.deepcopy(seen_primary)
            seen2 = copy.deepcopy(seen_primary)

            current_reward = buy_prob * (self.prices[primary, arms_pulled[primary]] - self.costs[primary]) * \
                             (1 + self.lam[user_class])
            reward_secondary_1 = buy_prob * self.P[primary, first_secondary, user_class] * \
                                 self.calculate_reward(seen1, first_secondary, arms_pulled, user_class)
            reward_secondary_2 = buy_prob * self.P[primary, second_secondary, user_class] * \
                                 self.calculate_reward(seen2, second_secondary, arms_pulled, user_class) * \
                                 self.lambda_secondary
            return current_reward + reward_secondary_1 + reward_secondary_2

    def calculate_total_reward(self, arms_pulled):
        tot_reward = 0
        for i in range(len(self.costs)):
            for user in range(len(self.lam)):
                tot_reward += self.alphas_par[i + 1] / (sum(self.alphas_par) - self.alphas_par[0]) * \
                              self.class_probability[user] * self.calculate_reward(
                    np.array([0, 0, 0, 0, 0]), i, arms_pulled, user)

        return tot_reward
