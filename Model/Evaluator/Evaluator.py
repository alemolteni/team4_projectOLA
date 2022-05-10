import numpy as np

class Evaluator:
    def __init__(self, products_list=[], click_prob_matrix=None, lambda_prob=0.5, conversion_rates=[], 
                alphas=[], margins=[], verbose=False):
        assert len(products_list) == len(conversion_rates) and len(products_list) == len(alphas)
        assert len(products_list) == len(margins)
        assert click_prob_matrix is not None

        self.click_prob_matrix = np.array(click_prob_matrix).tolist()
        self.products_list = products_list
        self.lambda_prob = lambda_prob
        self.conversion_rates = np.array(conversion_rates)
        self.n_products = len(products_list)
        self.alphas = np.array(alphas)
        self.margins = np.array(margins)
        self.verbose = verbose

    def computeMargin(self):
        return 0