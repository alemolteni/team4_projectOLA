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

    def computeMargin_per_class(self, arms):
        result = 0
        marginsPerPrice = self.config["margins"]
        if len(arms) == 1:
            return self.computeMargin(arms[0][0])
        else:
            for i in range(0, len(arms)):
                split_feature = arms[i][1]
                classes = self.config["classes"]
                for uc in classes:
                    feature_names = []
                    feature_values = []
                    for i in range(0, len(uc["features"])):
                        if uc["features"][i]["probability"] == "None":
                            uc["features"][i]["probability"] = None
                        feature_values.append(uc["features"][i]["probability"])
                        feature_names.append(uc["features"][i]["name"])
                    compliant = True
                    for f in split_feature:
                        for j in range(0, len(feature_names)):
                            if feature_names[j] == f:
                                idx = j
                        if feature_values[idx] != split_feature[f]:
                            compliant = False
                    if compliant:
                        arm = arms[i][0]
                        margins = [marginsPerPrice[i][arm[i]] for i in range(0, len(arm))]
                        armConvRates = [uc["conversionRates"][i][arm[i]] for i in range(0, len(arm))]
                        product_list = [Product(int(key), uc["secondary"][key]) for key in uc["secondary"]]
                        eval = GraphEvaluator(products_list=product_list, click_prob_matrix=uc["clickProbability"],
                                              lambda_prob=uc["lambda"], conversion_rates=armConvRates,
                                              alphas=uc["alphas"], margins=margins, units_mean=uc["actualUnitsMean"],
                                              convert_units=False, verbose=False)
                        result += eval.computeMargin()*uc["usersMean"]
            return result / self.user_means.sum()