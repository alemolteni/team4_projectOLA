import numpy as np
import math
from Learner.TS import *
from Model.Evaluator.GraphEvaluator import *


class TS_CR(TS):

    def __init__(self, num_products=5, num_prices=4, margins=np.ones((5, 4)), alphas=np.ones(5), secondary_prod=[],
                 click_prob=np.ones((5, 5)), l=0.5, units_mean=np.ones(5), convert_units=False, debug=False):
        super(TS_CR, self).__init__(num_products=num_products, num_prices=num_prices)
        self.margins = margins
        self.alphas = alphas
        self.click_prob = click_prob
        self.l = l
        self.product_list = secondary_prod

        self.units_mean = units_mean
        if convert_units:
            actual_means = []
            for i in range(0,len(units_mean)):
                empiric_mean = np.ceil(np.random.gamma(units_mean[i], 1, size=1000000)).mean()
                actual_means.append(int(empiric_mean*100) / 100)
            self.units_mean = actual_means

        # Take as input also alphas or other information known needed for computing expected rewards
        # in this class only conversion rates are unknown

        # Look at super class TS to see ho it works, but briefly we use 
        # self.conversion_rates_distro to save parameters alpha and beta of the distribution
        # then they are used to sample a conversion rate for each arm

    def pull_arm(self):
        return super(TS_CR, self).pull_arm()

    def compute_expected_rewards(self):
        # It should return a matrix #PROD x #LEVELS in which the elements are the computed rewards
        # Then pull arm will use this to choose the arm with the max expected reward as next
        exp_rewards = np.zeros((self.num_products, self.num_prices))

        for i in range(0, self.num_products):
            for j in range(0, self.num_prices):
                test_config = self.configuration
                test_config[i] = j
                margin = self.compute_product_margin(test_config)
                exp_rewards[i, j] = margin

        return exp_rewards

    def compute_product_margin(self, test_config):
        armMargins = []
        armConvRates = []
        for k in range(0, len(test_config)):
            armMargins.append(self.margins[k][test_config[k]])
            armConvRates.append(self.used_conv_rates[k][test_config[k]])
            # armConvRates.append(np.random.beta(self.conversion_rates_distro[k][self.configuration[k]][0], self.conversion_rates_distro[k][self.configuration[k]][1]))

        graphEval = GraphEvaluator(products_list=self.product_list, click_prob_matrix=self.click_prob,
                                   lambda_prob=self.l, alphas=self.alphas, conversion_rates=armConvRates,
                                   margins=armMargins,
                                   units_mean=self.units_mean, verbose=False, convert_units=False)
        margin = graphEval.computeMargin()
        return margin

    def update(self, interactions):
        # From daily interactions extract needed information, depending on step uncertainty:
        #   - Step 3: update belief over conversion rates
        #   - Step 4: update conversion rates, 𝛼 ratios, #units sold per product
        #   - Step 5: update conversion rates, graph weights
        super(TS_CR, self).update(interactions)
        return 
        
