import numpy as np
from EnvironmentPricing import EnvironmentPricing
from scipy.stats import norm


class EnvironmentPricingAggregated(EnvironmentPricing):
    def __init__(self, mean, variance, prices, costs, lambdas, alphas_par, P, secondary_products, lambda_secondary):
        super().__init__(mean, variance, prices, costs, lambdas, alphas_par, P, secondary_products, lambda_secondary)
        self.mean = np.mean(self.mean, axis=1)          # Expected value of reservation price for user (Aggregated)
        self.variance = np.mean(self.variance, axis=1)  # Variance of reservation price for user (Aggregated)

        self.lam = np.mean(self.lam)        # The number of items bought ~ 1 + Poisson(lam) (Aggregated)
        self.P = np.mean(self.P, axis=2)    # Click probability of the secondary product from a primary product (Aggregated)

        self.alpha_ratio_data = [[] for _ in range(5)] # List of list. One inner list for each product saving 1 if user ends up in that product page as first, 0 otherwise
        self.conv_data = [[] for _ in range(5)]  # List of list. One inner list for each product saving 1 if user buys when shown as primary, 0 otherwise
        self.number_objects_data = [[] for _ in range(5)]
        self.alpha_ratio = np.zeros(6)      # Daily value of the alpha ratios

    def round_single_day(self, n_daily_users, arms_pulled, to_find):
        self.alpha_ratio = self.alpha_ratio_otd()   #Initialize alphas
        self.alpha_ratio_data = [[] for _ in range(5)] #Initialize alpha ratio lists
        self.conv_data = [[] for _ in range(5)]     #Initialize conversion lists
        self.number_objects_data = [[] for _ in range(5)] #Initialize number of objects lists
        return_list = [[] for _ in range(len(to_find))]


        for u in range(0,n_daily_users):
            self.round_single_customer(arms_pulled)

        if "alphas" in to_find:
            return_list[0] = self.alpha_ratio_data
        if "conversion_rate" in to_find:
            if "alphas" in to_find:
                return_list[1] = self.conv_data
            else:
                return self.conv_data
        if "quantity" in to_find:
            return_list[2] = self.number_objects_data

        return return_list

    # Returns the reward of a single product bought
    def round_single_product(self, product, arm_pulled):
        mean = self.mean[product]
        var = self.variance[product]

        # if it goes below zero this means that the user doesn't buy the product selected
        reservation_price = np.random.normal(loc=mean, scale=np.sqrt(var))

        if self.prices[product, arm_pulled] <= reservation_price:
            self.conv_data[product].append(1)
            self.number_objects_data[product].append(np.random.poisson(lam=self.lam) + 1)
            return 1
        else:
            self.conv_data[product].append(0)
            return 0


    # Returns the reward of all the items bought by a single customer
    def round_single_customer(self, arms_pulled):
        seen_primary = np.full(shape=5, fill_value=False)
        current_product = np.random.choice(a=[-1, 0, 1, 2, 3, 4],
                                           p=self.alpha_ratio)  # CASE -1: the customer goes to a competitor

        for i in range(5):
            if i == current_product:
                self.alpha_ratio_data[i].append(1)
            else:
                self.alpha_ratio_data[i].append(0)

        if current_product == -1:
            return  # since the customer didn't visit our site, we don't consider him when learning

        seen_primary[current_product] = True
        self.round_recursive(seen_primary, current_product, arms_pulled)
        return


    # Auxiliary function needed in round_single_customer. Explore the tree in DFS
    def round_recursive(self, seen_primary, primary, arms_pulled):
        buyed = self.round_single_product(primary, arms_pulled[primary])

        if not buyed:
            return

        else:
            secondary_1 = self.secondary_products[primary, 0]
            secondary_2 = self.secondary_products[primary, 1]

            if not seen_primary[secondary_1]:
                click_slot_1 = np.random.binomial(n=1, p=self.P[
                    primary, secondary_1])  # clicks on the shown product to visualize its page
                if click_slot_1:
                    seen_primary[secondary_1] = True
                    self.round_recursive(seen_primary, secondary_1, arms_pulled)

            if not seen_primary[secondary_2]:
                p_ = self.P[primary, secondary_2] * self.lambda_secondary
                click_slot_2 = np.random.binomial(n=1,
                                                          p=p_)  # clicks on the shown product to visualize its page
                if click_slot_2:
                    seen_primary[secondary_2] = True
                    self.round_recursive(seen_primary, secondary_2, arms_pulled)


    def get_real_conversion_rates(self):
        conv_rate = np.zeros((5,4))
        for i in range(0, 5):
            for j in range(0, 4):
                conv_rate[i,j] = 1 - norm.cdf(self.prices[i,j], self.mean[i], np.sqrt(self.variance[i]))

        return conv_rate

    def get_alpha_param(self):
        return self.alphas_par/np.sum(self.alphas_par)
