import numpy as np
from Learner.TS import *
from Learner.TS_CR import *


class TS_Alphas(TS_CR):

    def __init__(self, num_products=5, num_prices=4, margins=np.ones((5, 4)), secondary_prod=[], l=0.5,
                 click_prob=np.ones((5, 5)), debug=False):
        super(TS_Alphas, self).__init__(num_products=num_products, num_prices=num_prices, margins=margins,
                                        click_prob=click_prob, l=l, secondary_prod=secondary_prod)
        self.beta_parameters_alphas = np.ones((num_products, 2))

    def update(self, interactions, pulledArm):
        # From daily interactions extract needed information, depending on step uncertainty:
        #   - Step 5: update graph weights

        # From daily interactions extract needed information, depending on step uncertainty:
        #   - Step 3: update belief over conversion rates
        #   - Step 4: update conversion rates, 𝛼 ratios, #units sold per product
        #   - Step 5: update conversion rates, 𝛼 ratios, #units sold per product, graph weights

        for i in interactions["episodes"]:
            if i.bought:
                reward = 1.0
            else:
                reward = 0.0
            self.beta_parameters[i.product, pulledArm[i.product], 0] = self.beta_parameters[i.product, pulledArm[
                i.product], 0] + reward
            self.beta_parameters[i.product, pulledArm[i.product], 1] = self.beta_parameters[i.product, pulledArm[
                i.product], 1] + 1.0 - reward

            for j in range(0, self.num_products):
                if j == i.product:
                    self.beta_parameters_alphas[j, 0] = self.beta_parameters_alphas[
                                                            j, 0] + 1.0
                else:
                    self.beta_parameters_alphas[j, 1] = self.beta_parameters_alphas[
                                                            j, 1] + 1.0
        self.update_conversion_rates()
        self.update_alphas()
        return

    def update_alphas(self):
        for i in range(0, self.num_products):
            self.alphas[i] = self.beta_parameters_alphas[i, 0] / (
                    self.beta_parameters_alphas[i, 0] + self.beta_parameters_alphas[i, 1])
        print("\nupdated alphas: ", self.alphas)
