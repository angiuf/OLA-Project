import numpy as np

class Environment_Pricing:
    #Initialize the environment with the probabilities of purchasing a product wrt the price selected
    def __init__(self, average, variance, prices, lambdas, alphas_par, P, secondary_products, lambdas_secondary):
        self.average = average #(5, 3) for each product and each class every user has a different average
        self.variance = variance #(5, 3) for each product and each class there is a different gaussian variance
        self.prices = prices #this matrix contains the prices arms for each product (5,4)
        self.lam = lambdas #expected products bought by each class
        self.alphas_par = alphas_par #Paramenters of the Dirichlet previously calculated from the expected values
        self.P = P #(5,5,3) click probability of the secondary product from a primary product for each class
        self.secondary_products = secondary_products #(5,2) the two secondary products for each product in order
        self.lambdas_secondary = lambdas_secondary #fixed probability

    def round_single_product(self, product, arm_pulled, extracted_class):
        reservation_price = np.random.normal(loc = self.average[product, extracted_class], scale= np.sqrt(self.variance[product, extracted_class])) #if it goes below zero this means that the user doesn't buy the product selected
        if self.prices[product, arm_pulled] <= reservation_price:
            number_objects = np.random.poisson(lam=self.lam[extracted_class]) + 1
            reward = self.prices[product, arm_pulled] * number_objects
            return round(reward, 2)
        else:
            return 0

    def alpha_ratioOTD(self):
        return np.random.dirichlet(self.alphas_par)

    def round_single_customer(self, alpha_ratio, arms_pulled, class_probability):
        seen_primary = np.full(shape=5, fill_value=False)
        extracted_class = np.random.choice(a=[0, 1, 2], p=class_probability)
        current_product = np.random.choice(a=[-1, 0, 1, 2, 3, 4], p=alpha_ratio) #CASE -1: the customer goes to a competitor

        if current_product == -1:
            return -1 #since the customer didn't visit our site, we don't consider him when learning

        seen_primary[current_product] = True
        return round(self.round_recursive(seen_primary, current_product, 0, extracted_class, arms_pulled), 2)

    def round_recursive(self, seen_primary, primary, reward_until_now, extracted_class, arms_pulled):
        reward = self.round_single_product(primary, arms_pulled[primary], extracted_class)

        if reward == 0:
            return reward_until_now

        else:
            reward_until_now += reward
            secondary_1 = self.secondary_products[primary, 0]
            secondary_2 = self.secondary_products[primary, 1]

            if seen_primary[secondary_1]==0:
                first_secondary = np.random.binomial(n=1, p=self.P[primary, secondary_1, extracted_class]) #clicks on the shown product to visualize its page
                if first_secondary==1:
                    seen_primary[secondary_1]=1
                    reward_until_now += self.round_recursive(seen_primary, secondary_1, reward_until_now, extracted_class, arms_pulled)

            if seen_primary[secondary_2]==0:
                p=self.P[primary, secondary_2, extracted_class]*self.lambdas_secondary
                second_secondary = np.random.binomial(n=1, p=p) #clicks on the shown product to visualize its page
                if second_secondary == 1:
                    seen_primary[secondary_2]=1
                    reward_until_now += self.round_recursive(seen_primary, secondary_2, reward_until_now, extracted_class, arms_pulled)

        return reward_until_now
