import numpy as np
import math
from Learner.UCB import *

class UCB_CR(UCB):

    def __init__(self, margins=np.ones((5, 4)), num_products=5, num_prices=4, debug=False, alphas=np.ones(5),
                 clickProbability=np.ones((5,5)), secondary=None, Lambda=1):
        if secondary is None:
            secondary = {0: [1, 2], 1: [2, 4], 2: [3, 4], 3: [4, 0], 4: [1, 3]} #Changed secondary sintax to avoid strings
        self.alphas = alphas
        self.clickProbability = clickProbability
        self.secondary = secondary
        self.Lambda = Lambda
        super(UCB_CR, self).__init__(margins=margins, num_products=num_products, num_prices=num_prices, debug=debug)


        # Take as input also alphas or other information known needed for computing expected rewards
        # in this class only conversion rates are unknowns
        # Briefly in the super class we initialize the expected conversion rate 


    def pull_arm(self):
        # In practice it chooses arm with higher upper confidence bound
        # UCB is defined as arm ← argmax{a∈A} (x(a) + sqrt(2*log(t)/n(a,t − 1))
        # x(a) is the expected reward till now (which is the metric for the reward? to be defined in coumpute expected rew)

        return super(UCB_CR, self).pull_arm()

    def compute_expected_rewards(self):
        # It should return a matrix #PROD x #LEVELS in which the elements are the computed rewards
        # Then pull arm will use this to choose the arm with the max expected reward as next

        # Use estimated self.conversion_rates to compute the reward

        for i in range(0, len(self.configuration)):
                self.expected_reward[i][self.configuration[i]] = self.compute_expected_prod(i)

        return self.expected_reward

    def update(self, interactions):
        # From daily interactions extract needed information, depending from step uncertainty:
        #   - Step 3: update belief over conversion rates
        #   - Step 4: update conversion rates, 𝛼 ratios, #units sold per product
        #   - Step 5: update conversion rates, 𝛼 ratios, #units sold per product, graph weights

        # It already updates the conversion rates based on the interactions objects
        # In this step no other estimation is required

        super(UCB_CR, self).update(interactions)
        return

    def compute_expected_prod(self, prod):  # ToDO: add mean of product sold - gamma shape is the mean, rate = 1
        par = 0
        for i in range(0, len(self.configuration)):
               par += self.alphas[i] * self.compute_prob_from_a_to_b(i, prod)

        return par * self.margins[prod][self.configuration[prod]] * self.conversion_rates[prod][self.configuration[prod]]

    def compute_prob_from_a_to_b (self, a, b, trace=[]):
        if a == b: return 1

        trace.append(a)
        prob = 0
        prob2 = 0

        if self.secondary[a][0] not in trace:
            prob = self.clickProbability[a][self.secondary[a][0]] *\
                   self.conversion_rates[a][self.configuration[a]] *\
                   self.compute_prob_from_a_to_b(self.secondary[a][0], b, trace)

        if self.secondary[a][1] not in trace:
            prob2 = self.clickProbability[a][self.secondary[a][1]] * 0.8 * \
                    self.conversion_rates[a][self.configuration[a]] *\
                    self.compute_prob_from_a_to_b(self.secondary[a][1], b, trace)

        trace.pop()
        return prob + prob2