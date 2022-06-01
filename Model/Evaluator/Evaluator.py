import numpy as np
import math

class Evaluator:
    def __init__(self, products_list=[], click_prob_matrix=None, lambda_prob=0.5, conversion_rates=[], 
                alphas=[], margins=[], verbose=False, units_mean=[]):
        assert len(products_list) == len(conversion_rates) and len(products_list) == len(alphas)
        assert len(products_list) == len(margins)
        assert len(alphas) == len(units_mean)
        assert click_prob_matrix is not None

        self.click_prob_matrix = np.array(click_prob_matrix).tolist()
        self.products_list = products_list
        self.lambda_prob = lambda_prob
        self.conversion_rates = np.array(conversion_rates)
        self.n_products = len(products_list)
        self.alphas = np.array(alphas)
        self.margins = np.array(margins)
        actual_means = []
        # Cannot integrate the gamma function, so empirically compute the ceiled mean
        for i in range(0,len(units_mean)):
            empiric_mean = np.ceil(np.random.gamma(units_mean[i], 1, size=1000000)).mean()
            actual_means.append(int(empiric_mean*100) / 100)
        self.units_mean = np.array(actual_means)
        self.verbose = verbose

    def computeMargin(self):
        return 0