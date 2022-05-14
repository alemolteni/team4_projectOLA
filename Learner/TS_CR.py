import numpy as np
import math
import TS 


class TS_CR(TS):

    def __init__(self, num_products=5, num_prices=5, debug=False):
        super(TS_CR,self).__init__(num_products=num_products, num_prices=num_prices)
        # Take as input also alphas or other information known needed for computing expected rewards
        # in this class only converstion rates are unknows

        # Look at super class TS to see ho it works, but briefly we use 
        # self.conversion_rates_distro to save parameters alpha and beta of the distribution
        # then they are used to sample a conversion rate for each arm

    def pull_arm(self):
        return super(TS_CR,self).pull_arm()

    def compute_expected_rewards(self):
        # It should return a matrix #PROD x #LEVELS in which the elements are the computed rewards
        # Then pull arm will use this to choose the arm with the max expected reward as next
        return 

    def update(self, interactions):
        # From daily interactions extract needed informatio, depending from step uncertainty:
        #   - Step 3: update belief over conversion rates
        #   - Step 4: update conversion rates, 𝛼 ratios, #units sold per product
        #   - Step 5: update conversion rates, 𝛼 ratios, #units sold per product, graph weights

        # In this step we don't need to estimate other parameters, so super class can do all the work
        super(TS_CR,self).update(interactions)
        return 
