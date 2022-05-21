import numpy as np
import math


class UCB_CR():

    def __init__(self, margins=np.ones((5, 4)), num_products=5, num_prices=4, debug=False, alphas=np.zeros(5),
                 clickProbability=np.zeros((5, 5)), secondary=None, Lambda=1, conversion_rates=None):
        if secondary is None:
            secondary = {0: [1, 2], 1: [2, 4], 2: [3, 4], 3: [4, 0],
                         4: [1, 3]}  # Changed secondary sintax to avoid strings
        self.alphas = alphas
        self.clickProbability = clickProbability
        self.secondary = secondary
        self.Lambda = Lambda

        self.configuration = [0 for i in range(0, num_products)]
        self.t = 0
        self.debug = debug
        self.num_products = num_products
        self.num_prices = num_prices
        # Incremental average m(n+1) = m(n) + (new_val - m(n)) / n+1
        if conversion_rates is None:
            self.conversion_rates = np.zeros((self.num_products, self.num_prices))
        else:
            self.conversion_rates = conversion_rates
        self.times_arms_pulled = np.full((num_products, num_prices), 0.0)
        self.expected_reward = np.full((self.num_products, self.num_prices), 0.0)
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
        if self.debug: print("Reward: ", self.expected_reward)
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

        visits = np.full((self.num_products), 0)
        bought = np.full((self.num_products), 0)
        # print(interactions)
        for inter in interactions["episodes"]:
            visits = np.add(visits, inter.linearizeVisits())
            bought = np.add(bought, inter.linearizeBought())

        episode_conv_rates = np.divide(bought, visits, out=np.full_like(bought, 0, dtype=float), where=visits != 0)
        # print(episode_conv_rates)
        for i in range(0, len(self.configuration)):
            # Incremental average m(n+1) = m(n) + (new_val - m(n)) / n+1
            mean = self.conversion_rates[i][self.configuration[i]]
            self.conversion_rates[i][self.configuration[i]] = (mean + (episode_conv_rates[i] - mean) / self.t)
        if self.debug: print("Conversion rates: ", self.conversion_rates)
        return

    def compute_expected_rewards(self):
        # It should return a matrix #PROD x #LEVELS in which the elements are the computed rewards
        # Then pull arm will use this to choose the arm with the max expected reward as next

        # Use estimated self.conversion_rates to compute the reward

        for i in range(0, len(self.configuration)):
            self.expected_reward[i][self.configuration[i]] = self.compute_expected_prod(i)

        return self.expected_reward

    def compute_expected_prod(self, prod):  # ToDO: add mean of product sold - gamma shape is the mean, rate = 1
        par = 0
        for i in range(0, len(self.configuration)):
            par += self.alphas[i] * self.compute_prob_from_a_to_b(i, prod)

        return par * self.margins[prod][self.configuration[prod]] * self.conversion_rates[prod][
            self.configuration[prod]]

    def compute_prob_from_a_to_b(self, a, b, trace=[]):
        if a == b: return 1

        trace.append(a)
        prob = 0
        prob2 = 0

        if self.secondary[a][0] not in trace:
            prob = self.clickProbability[a][self.secondary[a][0]] * \
                   self.conversion_rates[a][self.configuration[a]] * \
                   self.compute_prob_from_a_to_b(self.secondary[a][0], b, trace)

        if self.secondary[a][1] not in trace:
            prob2 = self.clickProbability[a][self.secondary[a][1]] * 0.8 * \
                    self.conversion_rates[a][self.configuration[a]] * \
                    self.compute_prob_from_a_to_b(self.secondary[a][1], b, trace)

        trace.pop()
        return prob + prob2
