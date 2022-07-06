import numpy as np
import math
from Learner.TS import *
from Model.Evaluator.GraphEvaluator import *


class TS_GW(TS):

    def __init__(self, num_products=5, num_prices=4, margins=np.ones((5, 4)), alphas=np.ones(6), secondary_prod=[],
                 conversion_rates=np.ones((5, 4)), l=0.5, units_mean=[], debug=False):
        super(TS_GW, self).__init__(num_products=num_products, num_prices=num_prices)
        self.margins = margins
        self.alphas = alphas
        self.beta_parameters = np.ones((num_products, num_products, 2))
        self.mean_click_prob = np.zeros((num_products, num_products)) # used to check if they converge to the real probabilities
        self.sample_click_prob = np.zeros((num_products, num_products))
        self.conversion_rates = conversion_rates
        self.l = l
        self.last_pulled_config = [0, 0, 0, 0, 0]
        self.product_list = secondary_prod
        self.units_mean = units_mean

        self.draw_sample_click_prob()
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
        # It should return a matrix #PROD x #PRICE LEVELS in which the elements are the computed rewards
        # Then pull arm will use this to choose the arm with the max expected reward as next
        exp_rewards = np.zeros((self.num_products, self.num_products))
        for i in range(0, self.num_products):
            for j in range(0, self.num_prices):
                test_config = self.last_pulled_config
                test_config[i] = j
                margin = self.compute_product_margin(test_config)
                exp_rewards[i, j] = margin
                #exp_rewards[i, j] = self.conversion_rates[i][j] * \
                #                    self.margins[i][j] * self.compute_product_prob(i, test_config)
        return exp_rewards

    def compute_product_margin(self, test_config):
        armMargins = []
        armConvRates = []
        for k in range(0, len(test_config)):
            armMargins.append(self.margins[k][test_config[k]])
            armConvRates.append(self.conversion_rates[k][test_config[k]])

        graphEval = GraphEvaluator(products_list=self.product_list, click_prob_matrix=self.sample_click_prob,
                                   lambda_prob=self.l, alphas=self.alphas, conversion_rates=armConvRates,
                                   margins=armMargins,
                                   units_mean=self.units_mean, verbose=False, convert_units=False)
        margin = graphEval.computeMargin()
        return margin

    def update(self, interactions, pulledArm):
        # From daily interactions extract needed information, depending on step uncertainty:
        #   - Step 5: update graph weights

        single_interactions_list = []
        for i in interactions:
            single_interactions_list.append(i)

        while len(single_interactions_list) != 0:
            p = single_interactions_list.pop(0)

            # update
            if p.bought:
                # first element
                if p.sec1Opened:
                    reward = 1.0
                else:
                    reward = 0.0
                self.beta_parameters[p.product, p.firstSlot, 0] = self.beta_parameters[
                                                                      p.product, p.firstSlot, 0] + reward
                self.beta_parameters[p.product, p.firstSlot, 1] = self.beta_parameters[
                                                                      p.product, p.firstSlot, 1] + 1.0 - reward

                # second element
                if p.sec2Opened:
                    reward = 1.0
                else:
                    reward = 0.0
                self.beta_parameters[p.product, p.secondSlot, 0] = self.beta_parameters[
                                                                       p.product, p.secondSlot, 0] + reward
                self.beta_parameters[p.product, p.secondSlot, 1] = self.beta_parameters[
                                                                       p.product, p.secondSlot, 1] + 1.0 - reward

            # expand
            for j in range(0, len(p.following)):
                single_interactions_list.append(p.following[j])

        self.draw_sample_click_prob()
        return

    def get_mean_click_prob(self):
        init = False
        for i in range(0, self.num_products):
            for j in range(0, self.num_products):
                if init and (self.product_list[i].getSecondaryProduct(0) == j or self.product_list[i].getSecondaryProduct(1) == j):
                    self.mean_click_prob[i, j] = 0.5
                elif self.product_list[i].getSecondaryProduct(0) == j or self.product_list[i].getSecondaryProduct(1) == j:  # if is a secondary product
                    self.mean_click_prob[i, j] = self.beta_parameters[i, j, 0] / (
                            self.beta_parameters[i, j, 0] + self.beta_parameters[i, j, 1])
                    if self.product_list[i].getSecondaryProduct(1) == j:  # if is the second secondary product
                        self.mean_click_prob[i, j] = self.mean_click_prob[i, j] / self.l
        return self.mean_click_prob

    def draw_sample_click_prob(self):
        for i in range(0, self.num_products):
            for j in range(0, self.num_products):
                self.sample_click_prob[i, j] = np.random.beta(self.beta_parameters[i, j, 0], self.beta_parameters[i, j, 1])
                if self.product_list[i].getSecondaryProduct(0) == j or self.product_list[i].getSecondaryProduct(1) == j:  # if is a secondary product
                    self.sample_click_prob[i, j] /= self.l

    def compute_product_prob(self, prod, test_config):
        armMargins = []
        armConvRates = []
        for k in range(0, len(test_config)):
            armMargins.append(self.margins[k][test_config[k]])
            armConvRates.append(self.conversion_rates[k][test_config[k]])

        graphEval = GraphEvaluator(products_list=self.product_list, click_prob_matrix=self.sample_click_prob,
                                   lambda_prob=self.l, alphas=self.alphas, conversion_rates=armConvRates,
                                   margins=armMargins,
                                   units_mean=self.units_mean, verbose=False)
        product_prob = graphEval.computeSingleProduct(prod)

        probability = self.alphas[prod]
        for i in range(0, self.num_products):
            if i != prod:
                probability += self.alphas[i] * self.conversion_rates[i][
                    test_config[i]] * product_prob[i]

        return probability
