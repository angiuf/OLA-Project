import numpy as np
from Environment_Pricing import *

# Model is a dictionary containing real or estimated parameters:
# Conversion rates
# Alpha ratios
# Mean number of purchased products
# Secondary product
# P
# Lambdas


#Calcola arm ottimale con il greedy algorithm
def optimization_algorithm(prices, n_products, n_arms, model, verbose=False):
    verboseprint = print if verbose else lambda *a, **k,: None
    price_arm = np.zeros(n_products).astype('int') # These are the indeces of the selected price arm
    rewards = np.zeros(n_products)
    extr_prices = prices[range(n_products), price_arm]
    extr_conversion_rate = model["conversion_rate"][range(n_products), price_arm]
    act_rate = MC_simulation(model, extr_conversion_rate, n_products)
    initial_reward = return_reward(model, extr_prices, extr_conversion_rate, act_rate)
    prec_reward = initial_reward
    verboseprint('Initial reward: ', initial_reward)
    while True:
        max_arms_counter = 0
        for i in range(n_products):
            if price_arm[i] == n_arms-1:
                rewards[i] = 0
                max_arms_counter += 1
            else:
                add_price = np.zeros(n_products).astype('int')
                add_price[i] = 1
                extr_prices = prices[range(n_products), price_arm+add_price]
                act_rate = MC_simulation(model, extr_conversion_rate, n_products)
                rewards[i] = return_reward(model, extr_prices, extr_conversion_rate, act_rate)
                verboseprint("Reward of arm: ", price_arm + add_price, "is: ", rewards[i])

        if max_arms_counter == n_products:
            return price_arm

        idx = np.argmax(rewards)

        if rewards[idx] <= prec_reward:
            verboseprint('Final arm chosen: ', price_arm)
            return price_arm
        else:
            add_price = np.zeros(n_products).astype('int')
            add_price[idx] = 1
            price_arm = price_arm + add_price
            prec_reward = rewards[idx]
            verboseprint('Selected amr: ', price_arm, 'with reward: ', rewards[idx])


def return_reward(model, extr_prices, extr_conversion_rate, act_prob):
    reward = 0
    n_prod = len(extr_prices)

    for i in range(n_prod):
        for j in range(n_prod):
            reward += model["alphas"][i+1]*act_prob[i,j]*extr_conversion_rate[j]*extr_prices[j]*model["quantity"]

    return reward


def MC_simulation(model, extr_conversion_rate, n_products):
    act_rates = np.zeros((n_products,n_products))
    K = 100 # Number of simulation for each seeds

    for i in range(n_products):   # Each iteration I take a different product as a seed (i)
        zetas = np.zeros(n_products)    # Zetas is the number of time I've seen a product
        for k in range(K):
            seen_primary = np.full(shape=5, fill_value=False)
            seen_primary[i] = True

            round_recursive(model, seen_primary, i, extr_conversion_rate)
            zetas[seen_primary] += 1

        act_rates[i,:] = zetas / K

    return act_rates

# Auxiliary function needed in round_single_customer. Explore the tree in DFS
def round_recursive(model, seen_primary, primary, extr_conversion_rate):
    if extr_conversion_rate[primary] > 1:
        buyed = True
    else:
        buyed = np.random.binomial(1, extr_conversion_rate[primary])

    if not buyed:
        return

    else:
        secondary_1 = model["secondary_products"][primary, 0]
        secondary_2 = model["secondary_products"][primary, 1]

        if not seen_primary[secondary_1]:
            click_slot_1 = np.random.binomial(n=1, p=model["P"][
                primary, secondary_1])  # clicks on the shown product to visualize its page
            if click_slot_1:
                seen_primary[secondary_1] = True
                round_recursive(model, seen_primary, secondary_1, extr_conversion_rate)

        if not seen_primary[secondary_2]:
            p_ = model["P"][primary, secondary_2] * model["lambda_secondary"]
            click_slot_2 = np.random.binomial(n=1,
                                                      p=p_)  # clicks on the shown product to visualize its page
            if click_slot_2:
                seen_primary[secondary_2] = True
                round_recursive(model, seen_primary, secondary_2, extr_conversion_rate)

