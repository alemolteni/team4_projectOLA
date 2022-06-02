import numpy as np
from Model.Evaluator.Evaluator import Evaluator


class Baseline(Evaluator):
    def __init__(self, products_list=[], click_prob_matrix=None, lambda_prob=0.5, conversion_rates=[], alphas=[], margins=[], units_mean=[], convert_units=True, verbose=False):
            super(Baseline, self).__init__(products_list=products_list, click_prob_matrix=click_prob_matrix, lambda_prob=lambda_prob, conversion_rates=conversion_rates, 
                alphas=alphas, margins=margins, units_mean=units_mean, convert_units=convert_units, verbose=verbose)

    def computeMargin(self):
        return np.multiply(np.multiply(np.multiply(self.margins,self.units_mean),self.conversion_rates),self.alphas).sum()
