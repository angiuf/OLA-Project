from Environment_Pricing.Environment_Pricing import Environment_Pricing
import numpy as np

def generate_prices(product_prices):
    prices = np.zeros((len(product_prices), 4))
    changing = np.array([-0.1, -0.05, 0.1, 0.05])
    for i in range(len(product_prices)):
        prices[i, :] = np.ones(len(changing))*product_prices[i] + np.ones(len(changing))*product_prices[i]*changing

    return prices

purchase_prob= np.random.rand(5,4,3)
prices= generate_prices(np.array([8, 3, 5, 4, 2]))
class_probability= np.array([0.4, 0.2, 0.4])
lambdas= np.array([1, 2, 3])
env1 = Environment_Pricing(purchase_prob, prices, class_probability, lambdas)

for i in range(10):
    round = env1.round(0, np.mod(i, 3))
    print("Reward:", round[0],  "\tClass:", round[1])
