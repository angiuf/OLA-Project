from Environment_Pricing.Environment_Pricing import Environment_Pricing
import numpy as np

def generate_prices(product_prices):
    prices = np.zeros((len(product_prices), 4))
    changing = np.array([-0.05, -0.1, 0.1, 0.05])
    for i in range(len(product_prices)):
        prices[i, :] = np.ones(len(changing))*product_prices[i] + np.ones(len(changing))*product_prices[i]*changing
    return prices

average = np.array([[9, 10, 7],
                   [3, 3, 2],
                   [4, 4, 5],
                   [3, 3.5, 3],
                   [1.5, 2, 2]])

variance = np.array([[1, 1, 1],
                   [0.5, 0.5, 0.5],
                   [0.5, 0.5, 0.5],
                   [0.5, 0.5, 0.5],
                   [0.5, 0.5, 0.5]])

prices= generate_prices(np.array([8, 3, 5, 4, 2]))

class_probability= np.array([0.4, 0.2, 0.4])

lambdas= np.array([1, 2, 3])

alphas_par = np.array([2, 3, 2, 1, 1, 1])

P=np.random.uniform(0.95, 1, size=(5, 5, 3))

secondary_products = np.array([[1, 4],
                              [0, 2],
                              [3, 0],
                              [2, 4],
                              [0, 1]])

env1 = Environment_Pricing(average, variance, prices, lambdas, alphas_par, P, secondary_products, lambdas_secondary=0.5)

#Test for one day and 10 customers, the arms pulled are the minimum
alpha_ratio = env1.alpha_ratioOTD()

for i in range(10):
    round = env1.round_single_customer(alpha_ratio, np.array([0, 0, 0, 0, 0]), class_probability)
    print("Reward:", round)
