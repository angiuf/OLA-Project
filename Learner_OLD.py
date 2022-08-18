import numpy as np
from scipy.stats import norm


class Learner():

    def __init(self):
        pass

    def calculate_reward(self, seen_primary, primary, arms_pulled, user_class, lambdas_n_objects, prices,
                         secondary_products, lambda_secondary, click_probabilities, res_price_means,
                         res_price_variances):
        if seen_primary[primary] == True:
            return 0
        else:
            seen_primary[primary] = True
            seen_primary_copy = seen_primary.copy()
            buy_mean = res_price_means[primary, user_class]
            buy_var = res_price_variances[primary, user_class]
            buy_prob = norm.cdf(prices[primary, arms_pulled[primary]], buy_mean, np.sqrt(buy_var))
            number_objects = lambdas_n_objects[user_class] + 1

            first_secondary = secondary_products[primary, 0]
            second_secondary = secondary_products[primary, 1]

            return buy_prob * prices[primary, arms_pulled[primary]] * number_objects + \
                   buy_prob * click_probabilities[primary, first_secondary, user_class] * \
                   self.calculate_reward(seen_primary_copy, first_secondary, arms_pulled, user_class, lambdas_n_objects,
                                         prices, secondary_products,
                                         lambda_secondary, click_probabilities, res_price_means, res_price_variances) + \
                   lambda_secondary * buy_prob * click_probabilities[primary, second_secondary, user_class] * \
                   self.calculate_reward(seen_primary_copy, second_secondary, arms_pulled, user_class,
                                         lambdas_n_objects, prices, secondary_products,
                                         lambda_secondary, click_probabilities, res_price_means, res_price_variances)

    def calculate_total_reward(self, arms_pulled, alphas, class_probability, lambdas_n_objects, prices,
                               secondary_products, lambda_secondary, click_probabilities, res_price_means,
                               res_price_variances):
        tot_reward = 0
        for i in range(0, len(alphas) - 1):
            for user in range(0, len(class_probability)):
                tot_reward += alphas[i + 1] * class_probability[user] * self.calculate_reward(
                    np.array([0, 0, 0, 0, 0]), i, arms_pulled, user, lambdas_n_objects, prices, secondary_products,
                    lambda_secondary, click_probabilities, res_price_means, res_price_variances)

        return tot_reward
