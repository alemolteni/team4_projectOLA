import numpy as np
import math


class TS:

    def __init__(self, num_products=5, num_prices=5, debug=False):
        self.configuration = [0 for i in range(0,num_products)] 
        self.t = 0
        self.debug = debug
        self.num_products = num_products
        self.num_prices = num_prices
        # Incremental average m(n+1) = m(n) + (new_val - m(n)) / n+1
        self.conversion_rates_distro = np.full((num_products, num_prices, 2), 1)
        self.used_conv_rates = np.full((self.num_products, self.num_prices), 0, dtype=float)
        self.configuration = np.zeros((num_products), dtype=int)
        self.times_arms_pulled = np.full((num_products, num_prices), 0.0)

    def pull_arm(self):
        # Choose arm with higher reward w.r.t. generated conversion rates
        self.t += 1
        for i in range(0, self.num_products):
            for j in range(0, self.num_prices):
                params = self.conversion_rates_distro[i][j]
                self.used_conv_rates[i][j] = np.random.beta(params[0], params[1])
        # print("Generated conv rates {}".format(self.used_conv_rates))
        # XXX must be the expected reward and must be computed for each product and price level
        self.expected_reward = self.compute_expected_rewards()
        self.configuration = np.argmax(self.expected_reward, axis=1)
        # print(self.expected_reward)
        return self.configuration

    def update(self, interactions):
        # From daily interactions extract needed informatio, depending from step uncertainty:
        #   - Step 3: update belief over conversion rates
        #   - Step 4: update conversion rates, 𝛼 ratios, #units sold per product
        #   - Step 5: update conversion rates, 𝛼 ratios, #units sold per product, graph weights
        
        visits = np.full((self.num_products),0)
        bought = np.full((self.num_products),0)
        for inter in interactions:
            visits = np.add(visits, inter.linearizeVisits())
            bought = np.add(bought, inter.linearizeBought())
        # Update distribution of the conv rates
        for i in range(0, self.num_products):
            self.times_arms_pulled[i][self.configuration[i]] += visits[i]
        for i in range(0,len(bought)):
            self.conversion_rates_distro[i][self.configuration[i]][0] += bought[i]
            self.conversion_rates_distro[i][self.configuration[i]][1] += (visits[i] - bought[i])
        return
