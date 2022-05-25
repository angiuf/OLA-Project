import numpy as np
from EnvironmentPricing import *


class EnvironmentPricingAggregated(EnvironmentPricing):
    def __init__(self, mean, variance, prices, costs, lambdas, alphas_par, P, secondary_products, lambda_secondary):
        super().__init__(mean, variance, prices, costs, lambdas, alphas_par, P, secondary_products, lambda_secondary)
        self.mean = np.mean(self.mean, axis=1)
        self.variance = np.mean(self.variance, axis=1)

        self.lam = np.mean(self.lam)
        self.P = np.mean(self.P, axis=2)


    def round_single_day(self, n_daily_users, arms_pulled):
        daily_reward = 0
        effective_users = 0
        alpha_ratio = self.alpha_ratio_otd()
        zetas = np.zeros((5,5))
        n_it_per_prod = np.zeros(5)

        for u in range(0,n_daily_users):
            reward_single_cust, seen, chosen_prod = self.round_single_customer(alpha_ratio, arms_pulled)
            if reward_single_cust != -1:
                daily_reward += reward_single_cust
                effective_users += 1
                zetas[chosen_prod, seen] += 1
                n_it_per_prod[chosen_prod] += 1

        activation_rates = np.zeros((5,5))
        for i in range(0,5):
            activation_rates[i,:] = zetas[i,:] / n_it_per_prod[i]

        print("Activation rates:\n", activation_rates)

        if effective_users == 0:
            return 0.0
        else:
            return daily_reward / n_daily_users, activation_rates

    # Returns the reward of a single product bought
    def round_single_product(self, product, arm_pulled):
        mean = self.mean[product]
        var = self.variance[product]

        # if it goes below zero this means that the user doesn't buy the product selected
        reservation_price = np.random.normal(loc=mean, scale=np.sqrt(var))

        if self.prices[product, arm_pulled] <= reservation_price:
            number_objects = np.random.poisson(lam=self.lam) + 1
            reward = (self.prices[product, arm_pulled] - self.costs[product]) * number_objects
            return round(reward, 2)
        else:
            return 0


    # Returns the reward of all the items bought by a single customer
    def round_single_customer(self, alpha_ratio, arms_pulled):
        seen_primary = np.full(shape=5, fill_value=False)
        current_product = np.random.choice(a=[-1, 0, 1, 2, 3, 4],
                                           p=alpha_ratio)  # CASE -1: the customer goes to a competitor


        if current_product == -1:
            return -1  # since the customer didn't visit our site, we don't consider him when learning

        seen_primary[current_product] = True
        print(seen_primary)
        print(current_product)
        return round(self.round_recursive(seen_primary, current_product, 0, arms_pulled), 2), seen_primary, current_product


    # Auxiliary function needed in round_single_customer
    def round_recursive(self, seen_primary, primary, reward_until_now, arms_pulled):
        reward = self.round_single_product(primary, arms_pulled[primary])

        if reward == 0:
            return reward_until_now

        else:
            reward_until_now += reward
            secondary_1 = self.secondary_products[primary, 0]
            secondary_2 = self.secondary_products[primary, 1]

            if not seen_primary[secondary_1]:
                buy_first_secondary = np.random.binomial(n=1, p=self.P[
                    primary, secondary_1])  # clicks on the shown product to visualize its page
                if buy_first_secondary:
                    seen_primary[secondary_1] = True
                    reward_until_now += self.round_recursive(seen_primary, secondary_1, reward_until_now
                                                             , arms_pulled)

            if not seen_primary[secondary_2]:
                p_ = self.P[primary, secondary_2] * self.lambda_secondary
                buy_second_secondary = np.random.binomial(n=1,
                                                          p=p_)  # clicks on the shown product to visualize its page
                if buy_second_secondary:
                    seen_primary[secondary_2] = True
                    reward_until_now += self.round_recursive(seen_primary, secondary_2, reward_until_now,
                                                            arms_pulled)

        return reward_until_now