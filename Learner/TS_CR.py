import numpy as np
import math
from Learner.TS import *
from Model.Evaluator.GraphEvaluator import *


class TS_CR(TS):

    def __init__(self, num_products=5, num_prices=4, margins=np.ones((5, 4)), alphas=np.ones(5), secondary_prod=[],
                 click_prob=np.ones((5, 5)), l=0.5, units_mean=[], debug=False):
        super(TS_CR, self).__init__(num_products=num_products, num_prices=num_prices)
        self.margins = margins
        self.alphas = alphas
        self.beta_parameters = np.ones((num_products, num_prices, 2))
        self.estimated_conversion_rates = np.ones((num_products, num_prices))
        self.click_prob = click_prob
        self.l = l
        self.last_pulled_config = [0, 0, 0, 0, 0]
        self.product_list = secondary_prod

        actual_means = []
        for i in range(0,len(units_mean)):
            empiric_mean = np.ceil(np.random.gamma(units_mean[i], 1, size=1000000)).mean()
            actual_means.append(int(empiric_mean*100) / 100)
        self.units_mean = actual_means
        # self.units_mean = units_mean

        self.update_conversion_rates()
        # Take as input also alphas or other information known needed for computing expected rewards
        # in this class only conversion rates are unknown

        # Look at super class TS to see ho it works, but briefly we use 
        # self.conversion_rates_distro to save parameters alpha and beta of the distribution
        # then they are used to sample a conversion rate for each arm

    def pull_arm(self):
        return super(TS_CR, self).pull_arm()
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
                margin = self.compute_product_margin(test_config)
                exp_rewards[i, j] = margin

        return exp_rewards

    def compute_product_margin(self, test_config):
        armMargins = []
        armConvRates = []
        for k in range(0, len(test_config)):
            armMargins.append(self.margins[k][test_config[k]])
            armConvRates.append(np.random.beta(self.beta_parameters[k, test_config[k], 0], self.beta_parameters[k, test_config[k], 1]))

        graphEval = GraphEvaluator(products_list=self.product_list, click_prob_matrix=self.click_prob,
                                   lambda_prob=self.l, alphas=self.alphas, conversion_rates=armConvRates,
                                   margins=armMargins,
                                   units_mean=self.units_mean, verbose=False, convert_units=False)
        margin = graphEval.computeMargin()
        return margin

    def update(self, interactions, pulledArm):
        super(TS_CR, self).update(interactions)
        return 
        # From daily interactions extract needed information, depending on step uncertainty:
        #   - Step 3: update belief over conversion rates
        #   - Step 4: update conversion rates, 𝛼 ratios, #units sold per product
        #   - Step 5: update conversion rates, 𝛼 ratios, #units sold per product, graph weights

        for i in interactions:
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
        armMargins = []
        armConvRates = []
        for k in range(0, len(test_config)):
            armMargins.append(self.margins[k][test_config[k]])
            armConvRates.append(self.estimated_conversion_rates[k][test_config[k]])

        graphEval = GraphEvaluator(products_list=self.product_list, click_prob_matrix=self.click_prob,
                                   lambda_prob=self.l, alphas=self.alphas, conversion_rates=armConvRates,
                                   margins=armMargins,
                                   units_mean=self.units_mean, verbose=False)
        product_prob = graphEval.computeSingleProduct(prod)

        probability = self.alphas[prod]
        for i in range(0, self.num_products):
            if i != prod:
                probability += self.alphas[i] * self.estimated_conversion_rates[i, test_config[i]] * product_prob[i]

        return probability
