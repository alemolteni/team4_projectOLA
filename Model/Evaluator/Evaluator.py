import numpy as np
import math

class Evaluator:
    def __init__(self, products_list=[], click_prob_matrix=None, lambda_prob=0.5, conversion_rates=[], 
                alphas=[], margins=[], verbose=False, units_mean=None, convert_units=True):
        assert len(products_list) == len(conversion_rates) and len(products_list) == len(alphas)
        assert len(products_list) == len(margins)
        assert len(alphas) == len(units_mean)
        assert click_prob_matrix is not None
        assert units_mean is not None

        self.click_prob_matrix = np.array(click_prob_matrix).tolist()
        self.products_list = products_list
        self.lambda_prob = lambda_prob
        self.conversion_rates = np.array(conversion_rates)
        self.n_products = len(products_list)
        self.alphas = np.array(alphas)
        self.margins = np.array(margins)

        # Cannot integrate the gamma function, so empirically compute the ceiled mean
        # In environment: num_units = math.ceil(gamma(expected_val))
        # So the average number of units sold is different from expected_val
        # To get actual mean we should integrate gamma between [0,1] to get P(units=1) and so on for all intervals
        # Then the actual units mean will be sum{i} of P(units=i)*i
        # I've done it empirically by generating a lot of samples (not so clean solution but effective)
        #       E.G. np.ceil(np.random.gamma(.95, 1, size=1000000)).mean() ---> mean = 1.541436
        #       If I repeat the process 10 times then std = 0.0011484
        #       It is much different from ceil(expected_val = 0.95) = 1 <> actual_mean = 1.54
        if convert_units:
            actual_means = []
            for i in range(0,len(units_mean)):
                empiric_mean = np.ceil(np.random.gamma(units_mean[i], 1, size=1000000)).mean()
                actual_means.append(int(empiric_mean*100) / 100)
            self.units_mean = np.array(actual_means)
        else:
            self.units_mean = np.array(units_mean)
            
        self.verbose = verbose

    def computeMargin(self):
        return 0