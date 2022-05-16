import numpy as np
import math


class UCB:

    def __init__(self, num_products=5, num_prices=4, debug=False, margins=np.ones((5, 4))):
        self.configuration = [0 for i in range(0,num_products)] 
        self.t = 0
        self.debug = debug
        self.num_products = num_products
        self.num_prices = num_prices
        # Incremental average m(n+1) = m(n) + (new_val - m(n)) / n+1
        self.conversion_rates = np.full((num_products, num_prices), 0.0)
        self.times_arms_pulled = np.full((num_products, num_prices), 0.0)
        self.expected_reward = np.full((self.num_products, self.num_prices), 1.0)
        self.margins = margins

    def pull_arm(self):
        # Choose arm with higher upper confidence bound
        # UCB is defined as arm ← argmax{a∈A} (x(a) + sqrt(2*log(t)/n(a,t − 1))
        # x(a) is the expected reward till now (which is the metric for the reward?)
        self.t += 1
        log_time = np.full((self.num_products, self.num_prices), 2*math.log(self.t), dtype=float)
        n_arms = self.times_arms_pulled
        upper_deviation = np.sqrt(np.divide(log_time, n_arms, out=np.full_like(log_time, np.inf, dtype=float), where=n_arms!=0))
        self.expected_reward = self.compute_expected_rewards()
        #print(self.expected_reward)
        upper_bound = np.add(self.expected_reward, upper_deviation)
        self.configuration = np.argmax(upper_bound, axis=1)
        for i in range(0,len(self.configuration)):
            self.times_arms_pulled[i][self.configuration[i]] += 1
        return self.configuration

    def update(self, interactions):
        # From daily interactions extract needed information, depending from step uncertainty:
        #   - Step 3: update belief over conversion rates
        #   - Step 4: update conversion rates, 𝛼 ratios, #units sold per product
        #   - Step 5: update conversion rates, 𝛼 ratios, #units sold per product, graph weights
        
        visits = np.full((self.num_products),0)
        bought = np.full((self.num_products),0)
        for inter in interactions:
            visits = np.add(visits, inter.linearizeVisits())
            bought = np.add(bought, inter.linearizeBought())
        episode_conv_rates = np.divide(bought, visits, out=np.full_like(bought, 0, dtype=float), where=visits!=0)
        #print(episode_conv_rates)
        for i in range(0, len(self.configuration)):
            # Incremental average m(n+1) = m(n) + (new_val - m(n)) / n+1
            mean = self.conversion_rates[i][self.configuration[i]]
            self.conversion_rates[i][self.configuration[i]] = (mean + (episode_conv_rates[i] - mean) / self.t)
        #print (self.conversion_rates)
        return

    def compute_expected_rewards(self):
        exp = np.multiply(self.margins, self.conversion_rates)
        if self.debug: print("exp: ", exp)
        return exp
