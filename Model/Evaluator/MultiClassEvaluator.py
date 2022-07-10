import numpy as np
import json
from Model.Evaluator.GraphEvaluator import GraphEvaluator
from scipy.stats import gamma
from Model.Product import *
import math

class MultiClassEvaluator:
    def __init__(self, config_path=None):
        assert config_path is not None

        f = open(config_path)
        self.config = json.load(f)
        f.close()

        marginsPerPrice = self.config["margins"]
        self.num_prods = len(marginsPerPrice)
        self.num_prices = marginsPerPrice[0]

        user_means = []
        for uc in self.config["classes"]:
            user_means.append(uc["usersMean"])
        self.user_means = np.array(user_means)        


    def computeMargin(self, arm):
        marginsPerPrice = self.config["margins"]
        margins = [marginsPerPrice[i][arm[i]] for i in range(0,len(arm))]

        conf_classes = self.config["classes"]
        evaluators_results = []
        for uc in conf_classes:
            armConvRates = [uc["conversionRates"][i][arm[i]] for i in range(0,len(arm))]
            product_list = [Product(int(key), uc["secondary"][key]) for key in uc["secondary"]]
            eval = GraphEvaluator(products_list=product_list, click_prob_matrix=uc["clickProbability"], lambda_prob=uc["lambda"], conversion_rates=armConvRates,
                        alphas=uc["alphas"], margins=margins, units_mean=uc["actualUnitsMean"], convert_units=False, verbose=False)
            evaluators_results.append(eval.computeMargin())
        evaluators_results = np.array(evaluators_results)

        return np.multiply(self.user_means, evaluators_results).sum() / self.user_means.sum()