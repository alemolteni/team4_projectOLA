import numpy as np
from Learner.TS import *
from Learner.TS_CR import *


class TS_Alphas(TS_CR):

    def __init__(self, num_products=5, num_prices=4, margins=np.ones((5, 4)), secondary_prod=[], l=0.5,
                 click_prob=np.ones((5, 5)), debug=False):
        super(TS_Alphas, self).__init__(num_products=num_products, num_prices=num_prices, margins=margins,
                                        click_prob=click_prob, l=l, secondary_prod=secondary_prod)
        self.starting_cumulative = np.full(self.num_products, 0)
        self.units_cumulative = np.full(self.num_products, 0, dtype=float)
        self.total_times_bought = np.full(self.num_products, 0, dtype=float)

        self.alphas = np.full(self.num_products, 1/self.num_products)
        self.units_mean = np.full(self.num_products, 1)

    # alphas parameters are treated as the mean of a beta distribution

    def update(self, interactions):
        # From daily interactions extract needed information, depending on step uncertainty:
        #   - Step 4: update conversion rates, 𝛼 ratios, #units sold per product

        super(TS_Alphas, self).update(interactions)

        for inter in interactions:
            self.starting_cumulative = np.add(self.starting_cumulative, inter.linearizeStart())
            self.units_cumulative = np.add(self.units_cumulative, inter.linearizeNumUnits())
            self.total_times_bought = np.add(self.total_times_bought, inter.linearizeBought())
        
        self.alphas = self.starting_cumulative / self.starting_cumulative.sum()
        
        # https://stackoverflow.com/questions/26248654/how-to-return-0-with-divide-by-zero
        # c = np.divide(a, b, out=np.full_like(a, 1), where=b!=0)
        self.units_mean = np.divide(self.units_cumulative, self.total_times_bought, 
                                    out=np.full_like(self.units_cumulative, 1), where=self.total_times_bought!=0)
        return
