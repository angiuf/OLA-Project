import numpy as np
from scipy.stats import norm
from Source.EnvironmentPricing import EnvironmentPricing


class NonStationaryEnvironment(EnvironmentPricing):
    # In this class mean and variance have size (n_phases, 5,3)
    def __init__(self, mean, variance, prices, costs, lambdas, alphas_par, P, secondary_products, class_probability, lambda_secondary,
                 horizon):
        super().__init__(mean, variance, prices, costs, lambdas, alphas_par, P, secondary_products, class_probability, lambda_secondary)
        self.t = 0
        n_phases = self.mean.shape[2]
        self.phase_size = horizon / n_phases

    def round_single_day(self, n_daily_users, alpha_ratio, arms_pulled, class_probability):
        current_phase = int(self.t / self.phase_size)
        self.t += 1
        data = []
        for u in range(0, n_daily_users):
            data.append(self.round_single_customer(alpha_ratio, arms_pulled, class_probability, current_phase))
        return data

    def round_single_customer(self, alpha_ratio, arms_pulled, class_probability, current_phase):
        seen_primary = np.full(shape=5, fill_value=False)
        extracted_class = np.random.choice(a=[0, 1, 2], p=class_probability)
        current_product = np.random.choice(a=[-1, 0, 1, 2, 3, 4],
                                           p=alpha_ratio)  # CASE -1: the customer goes to a competitor

        number_objects = [0 for _ in range(5)]
        reward_per_object = [0 for _ in range(5)]
        bought_products = np.full(shape=5, fill_value=False)
        clicks = [[[] for _ in range(5)] for _ in range(5)]

        if current_product == -1:
            return [reward_per_object, number_objects, current_product, extracted_class, seen_primary, bought_products,
                    clicks]

        seen_primary[current_product] = True
        self.round_recursive(seen_primary, current_product, extracted_class, arms_pulled, number_objects,
                             reward_per_object, bought_products, clicks, current_phase)
        return [reward_per_object, number_objects, current_product, extracted_class, seen_primary, bought_products,
                clicks]

    # Auxiliary function needed in round_single_customer
    def round_recursive(self, seen_primary, primary, extracted_class, arms_pulled, number_objects,
                        reward_per_object, bought_products, clicks, current_phase):
        reward_per_object[primary], number_objects[primary] = self.round_single_product(primary, arms_pulled[primary],
                                                                                        extracted_class, current_phase)
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
                                         number_objects, reward_per_object, bought_products, clicks, current_phase)
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
                                         number_objects, reward_per_object, bought_products, clicks, current_phase)
                else:
                    clicks[primary][secondary_2].append(0)

        return

    # Returns the reward and the number of objects of a product bought
    def round_single_product(self, product, arm_pulled, extracted_class, current_phase):
        mean = self.mean[current_phase, product, extracted_class]
        var = self.variance[current_phase, product, extracted_class]

        # if it goes below zero this means that the user doesn't buy the product selected
        reservation_price = np.random.normal(loc=mean, scale=np.sqrt(var))

        if self.prices[product, arm_pulled] <= reservation_price:
            number_objects = np.random.poisson(lam=self.lam[extracted_class]) + 1
            reward = (self.prices[product, arm_pulled] - self.costs[product]) * number_objects
            return [round(reward, 2), number_objects]
        else:
            return [0, 0]

    def get_real_conversion_rates(self, class_, phase):
        conv_rate = np.zeros((5, 4))
        for i in range(0, 5):
            for j in range(0, 4):
                conv_rate[i, j] = 1 - norm.cdf(self.prices[i, j], self.mean[phase, i, class_],
                                               np.sqrt(self.variance[phase, i, class_]))
        return conv_rate
