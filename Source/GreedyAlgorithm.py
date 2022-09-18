from Source.EnvironmentPricing import *
from Source.Auxiliary import *


# Model is a dictionary containing real or estimated parameters:
# number of products,
# number of prices,
# prices,
# costs,
# alpha parameters,
# real conversion rates,
# mean number of purchased products,
# secondary product,
# P,
# lambda secondary,
# activation rates for each super arm are saved to avoid computing the again


# Runs the optimization algorithm to find the best arm, rates is the string name of what is used as conversion rate,
# e.g. to find the optimal arm we use the real conversion rates, in ucb we use means + widths, if act_rates is true
# use the already computed activation rates
def optimization_algorithm(model, env=None, verbose=False, rates="real_conversion_rates", alphas="real_alpha_ratio",
                           quantity="real_quantity", clicks='real_P', phase=-1):
    verbose_print = print if verbose else lambda *a, **k,: None
    n_prod = model["n_prod"]
    n_price = model["n_price"]
    price = model["prices"]
    K = 50  # number of seeds for MC simulation
    if rates == "real_conversion_rates":
        real = True
    else:
        real = False

    price_arm = np.zeros(n_prod).astype('int')  # These are the indexes of the selected price arm
    rewards = np.zeros(n_prod)  # rewards of current arm  increased by one

    if real:
        initial_reward = return_reward2(model, env, price_arm)
        previous_reward = initial_reward

    else:
        extracted_prices = price[range(n_prod), price_arm]
        # In a stationary environment there aren't any phases so the value is set to -1
        if phase == -1:
            extracted_cr = model[rates][range(n_prod), price_arm]
        else:
            extracted_cr = model[rates][phase, range(n_prod), price_arm]
        extracted_alpha = model[alphas]
        extracted_quantity = model[quantity]

        act_rate = mc_simulation(model, extracted_cr, n_prod, K, clicks)
        initial_reward = return_reward(model, extracted_prices, extracted_cr, act_rate, extracted_alpha,
                                       extracted_quantity)
        previous_reward = initial_reward

    verbose_print('Initial reward: ', initial_reward)
    while True:
        max_arms_counter = 0
        for i in range(n_prod):
            if price_arm[i] == n_price - 1:
                rewards[i] = -1
                max_arms_counter += 1
            else:
                add_price = np.zeros(n_prod).astype('int')
                add_price[i] = 1

                if real:
                    rewards[i] = return_reward2(model, env, price_arm)
                else:
                    extracted_prices = price[range(n_prod), price_arm]
                    # In a stationary environment there aren't any phases so the value is set to -1
                    if phase == -1:
                        extracted_cr = model[rates][range(n_prod), price_arm]
                    else:
                        extracted_cr = model[rates][phase, range(n_prod), price_arm]
                    extracted_alpha = model[alphas]
                    extracted_quantity = model[quantity]

                    act_rate = mc_simulation(model, extracted_cr, n_prod, K, clicks)
                    rewards[i] = return_reward(model, extracted_prices, extracted_cr, act_rate, extracted_alpha,
                                               extracted_quantity)

                verbose_print("Reward of arm: ", price_arm + add_price, "is: ", rewards[i])

        if max_arms_counter == n_prod:
            return price_arm
        idx = np.argmax(rewards)

        if rewards[idx] <= previous_reward:
            verbose_print('Final arm chosen: ', price_arm)
            if real:
                return price_arm, previous_reward
            else:
                return price_arm
        else:
            add_price = np.zeros(n_prod).astype('int')
            add_price[idx] = 1
            price_arm = price_arm + add_price
            previous_reward = rewards[idx]
            verbose_print('Selected arm: ', price_arm, 'with reward: ', rewards[idx])


# Returns the expected reward that a customer can give to the ecommerce
def return_reward(model, extracted_prices, extracted_cr, act_prob, extracted_alphas, extracted_quantity):
    reward = 0
    n_prod = len(extracted_prices)

    for i in range(n_prod):
        for j in range(n_prod):
            reward += extracted_alphas[i + 1] * act_prob[i, j] * np.min([extracted_cr[j], 100]) * (
                    extracted_prices[j] - model["cost"][j]) * extracted_quantity

    return reward


def return_reward2(model, env, pulled_arm, K=50):
    rewards = []
    for _ in range(K):
        alpha_ratio = env.alpha_ratio_otd()
        data = env.round_single_day(model["daily_user"], alpha_ratio, pulled_arm)
        rewards.append(calculate_reward(data))
    np_rewards = np.array(rewards)
    return np.mean(np_rewards)


# Computes a montecarlo simulation to compute the activation rates
def mc_simulation(model, extracted_cr, n_products, K=100, clicks='real_P'):
    act_rates = np.zeros((n_products, n_products))
    # K = number of simulation for each seeds

    for i in range(n_products):  # Each iteration I take a different product as a seed (i)
        zetas = np.zeros(n_products)  # Zetas is the number of time I've seen a product
        for k in range(K):
            seen_primary = np.full(shape=5, fill_value=False)
            seen_primary[i] = True
            round_recursive(model, seen_primary, i, extracted_cr, clicks)
            zetas[seen_primary] += 1

        act_rates[i, :] = zetas / K
    return act_rates


# Auxiliary function needed in round_single_customer. Explore the tree in DFS
def round_recursive(model, seen_primary, primary, extracted_cr, clicks):
    if extracted_cr[primary] > 1:
        buy = True
    else:
        buy = np.random.binomial(1, extracted_cr[primary])

    if not buy:
        return

    else:
        secondary_1 = model["secondary_products"][primary, 0]
        secondary_2 = model["secondary_products"][primary, 1]

        if not seen_primary[secondary_1]:
            click_slot_1 = np.random.binomial(n=1, p=model[clicks][
                primary, secondary_1])  # clicks on the shown product to visualize its page
            if click_slot_1:
                seen_primary[secondary_1] = True
                round_recursive(model, seen_primary, secondary_1, extracted_cr, clicks)

        if not seen_primary[secondary_2]:
            p_ = model[clicks][primary, secondary_2] * model["lambda_secondary"]
            click_slot_2 = np.random.binomial(n=1,
                                              p=p_)  # clicks on the shown product to visualize its page
            if click_slot_2:
                seen_primary[secondary_2] = True
                round_recursive(model, seen_primary, secondary_2, extracted_cr, clicks)
