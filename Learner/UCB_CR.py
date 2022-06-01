import numpy as np
import math
from Model.Evaluator.GraphEvaluator import *


class UCB_CR():

    def __init__(self, margins=np.ones((5, 4)), num_products=5, num_prices=4, debug=False, alphas=np.zeros(5),
                 clickProbability=np.zeros((5, 5)), secondary=None, Lambda=1, conversion_rates=None, units_mean=None):
        if units_mean is None:
            units_mean = [1, 1, 1, 1, 1]
        if secondary is None:
            secondary = {0: [1, 2], 1: [2, 4], 2: [3, 4], 3: [4, 0],
                         4: [1, 3]}  # Changed secondary syntax to avoid strings

        self.alphas = alphas
        self.clickProbability = clickProbability
        self.secondary = []
        self.productList = secondary
        for p in secondary:
            self.secondary.append([p.getSecondaryProduct(0), p.getSecondaryProduct(1)])

        self.Lambda = Lambda
        self.configuration = [0 for i in range(0, num_products)]
        self.t = 0
        self.debug = debug
        self.num_products = num_products
        self.num_prices = num_prices

        if conversion_rates is None:
            self.conversion_rates = np.zeros((self.num_products, self.num_prices))
        else:
            self.conversion_rates = conversion_rates
        self.times_arms_pulled = np.full((num_products, num_prices), 0.0)
        self.expected_reward = np.full((self.num_products, self.num_prices), 0.0)
        self.margins = margins
        self.units_mean = units_mean

        self.S = 1
        self.upper_bound = self.conversion_rates

    def pull_arm(self):
        # Choose arm with higher upper confidence bound
        # UCB is defined as arm ← argmax{a∈A} (x(a) + sqrt(2*log(t)/n(a,t − 1))
        # x(a) is the expected reward till now (which is the metric for the reward?)
        self.t += 1
        log_time = np.full((self.num_products, self.num_prices), 2 * math.log(self.t), dtype=float)
        n_arms = self.times_arms_pulled
        upper_deviation = np.sqrt(
            np.divide(log_time, n_arms, out=np.full_like(log_time, np.inf, dtype=float), where=n_arms != 0))

        self.expected_reward = self.compute_expected_rewardsGE()
        # Scale down the expected reward?
        # if np.amax(self.expected_reward) > 0:
        #    self.expected_reward = np.divide(self.expected_reward, np.amax(self.expected_reward))
        #print("ExpRew", self.expected_reward)

        # Scale up the upper_deviation element? For now has better results
        upper_deviation = np.multiply(upper_deviation, np.amin(self.expected_reward))

        self.upper_bound = np.add(self.expected_reward, upper_deviation)
        # upper_bound = np.clip(upper_bound, 0, 1)
        #print("UpperB", self.upper_bound)

        if self.t <= 4:
            self.configuration = [self.t - 1, self.t - 1, self.t - 1, self.t - 1, self.t - 1]
        else:
            self.configuration = np.argmax(self.upper_bound, axis=1)
        if self.debug:
            print("Config: ", self.configuration)
        for i in range(0, len(self.configuration)):
            self.times_arms_pulled[i][self.configuration[i]] += 1
        return self.configuration

    def update(self, interactions):
        # From daily interactions extract needed information, depending on step uncertainty:
        #   - Step 3: update belief over conversion rates

        visits = np.full(self.num_products, 0)
        bought = np.full(self.num_products, 0)

        for inter in interactions:
            visits = np.add(visits, inter.linearizeVisits())
            bought = np.add(bought, inter.linearizeBought())

        episode_conv_rates = np.divide(bought, visits, out=np.full_like(bought, 0, dtype=float), where=visits != 0)

        for i in range(0, len(self.configuration)):
            # Incremental average m(n+1) = m(n) + (new_val - m(n)) / n+1
            mean = self.conversion_rates[i][self.configuration[i]]
            self.conversion_rates[i][self.configuration[i]] = (mean + (episode_conv_rates[i] - mean) /
                                                               self.times_arms_pulled[i][self.configuration[i]])
        if self.debug:
            print("Conversion rates: ", self.conversion_rates)
        return

    def compute_expected_rewardsGE(self):
        # Graph evaluator tiene conto del CR nella probabilità di andare da uno all'altro?
        # Use graph evaluator to compute expected values

        armMargins = []
        armConvRates = []

        for k in range(0, len(self.configuration)):
            armMargins.append(self.margins[k][self.configuration[k]])
            armConvRates.append(self.conversion_rates[k][self.configuration[k]])

        graphEval = GraphEvaluator(products_list=self.productList, click_prob_matrix=self.clickProbability,
                                   lambda_prob=self.Lambda,
                                   alphas=self.alphas, conversion_rates=armConvRates, margins=armMargins,
                                   units_mean=self.units_mean, verbose=False)

        for i in range(0, len(self.configuration)):
            visiting_prob = graphEval.computeSingleProduct(i)
            visiting_prob = np.multiply(visiting_prob, self.alphas).sum()

            self.expected_reward[i][self.configuration[i]] = self.margins[i][self.configuration[i]] * \
                                                             self.conversion_rates[i][self.configuration[i]] * \
                                                             self.units_mean[i] * visiting_prob

        # print("Before config value: ", graphEval.computeMargin())
        return self.expected_reward


    # I'm not using methods below this point

    def compute_expected_rewards(self):
        # It should return a matrix #PROD x #LEVELS in which the elements are the computed rewards
        # Then pull arm will use this to choose the arm with the max expected reward as next

        # Use estimated self.conversion_rates to compute the reward

        for i in range(0, len(self.configuration)):
            self.expected_reward[i][self.configuration[i]] = self.compute_expected_prod(i)

        return self.expected_reward
    def compute_expected_prod(self, prod):
        par = 0
        for i in range(0, len(self.configuration)):
            par += self.alphas[i] * self.compute_prob_from_a_to_b(i, prod)

        return par * self.margins[prod][self.configuration[prod]] * self.conversion_rates[prod][self.configuration[prod]] * \
               self.units_mean[i]

    def compute_prob_from_a_to_b(self, a, b, trace=[]):
        if a == b:
            return 1

        trace.append(a)
        prob = 0
        prob2 = 0

        if self.secondary[a][0] not in trace:
            prob = self.clickProbability[a][self.secondary[a][0]] * \
                   self.conversion_rates[a][self.configuration[a]] * \
                   self.compute_prob_from_a_to_b(self.secondary[a][0], b, trace)

        if self.secondary[a][1] not in trace:
            prob2 = self.clickProbability[a][self.secondary[a][1]] * self.Lambda * \
                    self.conversion_rates[a][self.configuration[a]] * \
                    self.compute_prob_from_a_to_b(self.secondary[a][1], b, trace)

        trace.pop()
        return prob + prob2
