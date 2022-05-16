import numpy as np
import math
from Learner.TS import *


class TS_CR(TS):

    def __init__(self, num_products=5, num_prices=4, margins=np.ones((5, 4)), alphas=np.ones(6), debug=False):
        super(TS_CR, self).__init__(num_products=num_products, num_prices=num_prices)
        self.margins = margins
        self.alphas = alphas
        self.beta_parameters = np.ones((num_products, num_prices, 2))
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
            
        return pulled_config

    def compute_expected_rewards(self):
        # It should return a matrix #PROD x #LEVELS in which the elements are the computed rewards
        # Then pull arm will use this to choose the arm with the max expected reward as next
        exp_rewards = np.zeros((self.num_products, self.num_prices))
        for i in range(0, self.num_products):
            for j in range(0, self.num_prices):
                exp_rewards[i, j] = np.random.beta(self.beta_parameters[i, j, 0], self.beta_parameters[i, j, 1]) * self.margins[i][j]
        return exp_rewards

    def update(self, interactions, pulledArm):
        # From daily interactions extract needed information, depending on step uncertainty:
        #   - Step 3: update belief over conversion rates
        #   - Step 4: update conversion rates, 𝛼 ratios, #units sold per product
        #   - Step 5: update conversion rates, 𝛼 ratios, #units sold per product, graph weights

        for i in interactions:
            if i.bought:
                reward = 1.0
            else:
                reward = 0.0
            self.beta_parameters[i.product, pulledArm[i.product], 0] = self.beta_parameters[i.product, pulledArm[i.product], 0] + reward
            self.beta_parameters[i.product, pulledArm[i.product], 1] = self.beta_parameters[i.product, pulledArm[i.product], 1] + 1.0 - reward
        return
