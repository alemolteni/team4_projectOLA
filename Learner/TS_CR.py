import numpy as np
import math
from Learner.TS import *


class TS_CR(TS):

    def __init__(self, num_products=5, num_prices=4, margins=np.ones((5, 4)), alphas=np.ones(5), secondary_prod=[],
                 click_prob=np.ones((5, 5)), l=0.5, debug=False):
        super(TS_CR, self).__init__(num_products=num_products, num_prices=num_prices)
        self.margins = margins
        self.alphas = alphas
        self.beta_parameters = np.ones((num_products, num_prices, 2))
        self.estimated_conversion_rates = np.ones((num_products, num_prices))
        self.click_prob = click_prob
        self.l = l
        self.last_pulled_config = [0, 0, 0, 0, 0]
        self.secondary_prod = []
        for p in secondary_prod:
            self.secondary_prod.append([p.getSecondaryProduct(0), p.getSecondaryProduct(1)])

        self.update_conversion_rates()
        # Take as input also alphas or other information known needed for computing expected rewards
        # in this class only conversion rates are unknown

        # Look at super class TS to see ho it works, but briefly we use 
        # self.conversion_rates_distro to save parameters alpha and beta of the distribution
        # then they are used to sample a conversion rate for each arm

    def pull_arm(self):
        pulled_config = [0, 0, 0, 0, 0]
        exp_rewards = self.compute_expected_rewards()
        for i in range(0, self.num_products):
            pulled_config[i] = np.argmax(exp_rewards[i])

        self.last_pulled_config = pulled_config
        return pulled_config

    def compute_expected_rewards(self):
        # It should return a matrix #PROD x #LEVELS in which the elements are the computed rewards
        # Then pull arm will use this to choose the arm with the max expected reward as next
        exp_rewards = np.zeros((self.num_products, self.num_prices))
        for i in range(0, self.num_products):
            for j in range(0, self.num_prices):
                test_config = self.last_pulled_config
                test_config[i] = j
                exp_rewards[i, j] = np.random.beta(self.beta_parameters[i, j, 0], self.beta_parameters[i, j, 1]) * \
                                    self.margins[i][j] * self.compute_product_prob(i, test_config)
        return exp_rewards

    def update(self, interactions, pulledArm):
        # From daily interactions extract needed information, depending on step uncertainty:
        #   - Step 3: update belief over conversion rates
        #   - Step 4: update conversion rates, 𝛼 ratios, #units sold per product
        #   - Step 5: update conversion rates, 𝛼 ratios, #units sold per product, graph weights

        for i in interactions["episodes"]:
            if i.bought:
                reward = 1.0
            else:
                reward = 0.0
            self.beta_parameters[i.product, pulledArm[i.product], 0] = self.beta_parameters[
                                                                           i.product, pulledArm[i.product], 0] + reward
            self.beta_parameters[i.product, pulledArm[i.product], 1] = self.beta_parameters[i.product, pulledArm[
                i.product], 1] + 1.0 - reward

        self.update_conversion_rates()
        return

    def update_conversion_rates(self):
        for i in range(0, self.num_products):
            for j in range(0, self.num_prices):
                self.estimated_conversion_rates[i, j] = self.beta_parameters[i, j, 0] / (
                        self.beta_parameters[i, j, 0] + self.beta_parameters[i, j, 1])

    def compute_product_prob(self, prod, test_config):
        probability = self.alphas[prod]
        for i in range(0, self.num_products):
            if i != prod:
                probability += self.alphas[i] * self.estimated_conversion_rates[i, test_config[i]] * self.compute_prob_from_a_to_b(i, prod, test_config)

        return probability

    def compute_prob_from_a_to_b(self, a, b, test_config, trace=[]):
        if a == b: return 1

        trace.append(a)
        # print(trace)

        prob = 0
        prob2 = 0

        if self.secondary_prod[a][0] not in trace:
            prob = self.click_prob[a][self.secondary_prod[a][0]] * \
                   self.compute_prob_from_a_to_b(self.secondary_prod[a][0], b, test_config, trace) * \
                   self.estimated_conversion_rates[a, test_config[a]]
            # print("Prob1: ", prob)

        if self.secondary_prod[a][1] not in trace:
            prob2 = self.click_prob[a][self.secondary_prod[a][1]] * self.l * \
                    self.compute_prob_from_a_to_b(self.secondary_prod[a][1], b, test_config, trace) * \
                    self.estimated_conversion_rates[a, test_config[a]]
            # print("Prob2: ", prob2)

        trace.pop()
        return prob + prob2
