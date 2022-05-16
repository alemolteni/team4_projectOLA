import numpy as np
import math
from Learner.UCB import *

class UCB_CR(UCB):

    def __init__(self, margins=np.ones((5, 4)), num_products=5, num_prices=4, debug=False, alphas=np.ones(6)):
        self.alphas = alphas
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

        return super(UCB_CR, self).compute_expected_rewards()

    def update(self, interactions):
        # From daily interactions extract needed information, depending from step uncertainty:
        #   - Step 3: update belief over conversion rates
        #   - Step 4: update conversion rates, 𝛼 ratios, #units sold per product
        #   - Step 5: update conversion rates, 𝛼 ratios, #units sold per product, graph weights
        
        # It already updates the conversion rates based on the interactions objects
        # In this step no other estimation is required

        super(UCB_CR, self).update(interactions)
        return
