import numpy as np
import math
from Learner.TS import *
from Learner.TS_CR import *

from Model.Evaluator.GraphEvaluator import *


class TS_GW(TS_CR):

    def __init__(self, num_products=5, num_prices=4, margins=np.ones((5, 4)), alphas=np.ones(6), secondary_prod=[],
                 conversion_rates=np.ones((5, 4)), l=0.5, units_mean=[], debug=False):
        super(TS_GW, self).__init__(num_products=num_products, num_prices=num_prices, margins=margins, alphas=alphas, 
                            secondary_prod=secondary_prod, l=l, units_mean=units_mean, convert_units=False, debug=False)
        self.mean_click_prob = np.zeros((num_products, num_products)) # used to check if they converge to the real probabilities
        self.sample_click_prob = np.zeros((num_products, num_products))

        # Times in which secondary product j is opened from product i
        self.opening_cumulative = np.zeros((num_products, num_products), dtype=float)
        self.max_possible_opening_cumulative = np.zeros((num_products, num_products), dtype=float)

        self.prob_seen_given_bought_inverted = np.zeros((num_products, num_products), dtype=float)
        
        for prod in secondary_prod:
            primary = prod.getProductNumber()
            sec_list = prod.getSecondaryProductList()
            self.prob_seen_given_bought_inverted[primary][sec_list[0]] = 1
            self.prob_seen_given_bought_inverted[primary][sec_list[1]] = 1 / l # this is lambda not one

    def update(self, interactions):
        # From daily interactions extract needed information, depending on step uncertainty:
        #   - Step 5: update graph weights
        super(TS_GW, self).update(interactions)

        for inter in interactions:
            self.opening_cumulative = np.add(self.opening_cumulative, inter.linearizeSecondaryOpening())
            self.max_possible_opening_cumulative = np.add(self.max_possible_opening_cumulative, inter.linearizePossibleSecondaryOpening())

        opening_prob_given_bought = np.divide(self.opening_cumulative, self.max_possible_opening_cumulative, 
                                out=np.full_like(self.opening_cumulative, 0), where=self.max_possible_opening_cumulative!=0)
        
        opening_prob_given_seen = opening_prob_given_bought * self.prob_seen_given_bought_inverted
        self.click_prob = opening_prob_given_seen
        return

