import numpy as np

#Each class has its own environment

class Environment_Pricing:
    #Initialize the environment with the probabilities of purchasing a product wrt the price selected
    def __init__(self, purchase_prob, prices, class_probability, lambdas):
        self.probs = purchase_prob #this is a matrix of shape (5,4,3) where in the rows we keep the products, in the columns the probability of purchasing with different prices in each class
        self.prices = prices #this matrix contains the prices arms for each product (5,4)
        self.class_probs = class_probability #our target distribution
        self.lam = lambdas #expected products bought by each class

    def round(self, product, arm_pulled):
        extracted_class = np.random.choice(a=[0, 1, 2], p = self.class_probs)
        conversion = np.random.binomial(n=1, p=self.probs[product, arm_pulled, extracted_class]) #1 if bought, 0 otherwise
        number_objects = np.random.poisson(lam=self.lam[extracted_class]) + 1
        reward = conversion * self.prices[product, arm_pulled] * number_objects
        return round(reward, 2), extracted_class