import numpy as np

from Environment import Environment
from Model.ConfigurationParametersAverage import mergeUserClasses


class ContextGeneration:

    def __init__(self, features=None, num_products=5):
        if features is None:
            self.features = ['', '']
        else:
            self.featuresNames = features

        self.num_products = num_products

        self.contexts_visits = []
        self.contexts_bought = []
        self.contexts_started = []
        self.contexts_units = []

    def computePartition(self, interactions, optimal_arm):
        # Since we have two binary variables, we have 2^2^2 = 16 contexts and 10 partitions

        # One feature context
        for i in [True, False]:
            for j in [True, False]:
                visits = np.full(self.num_products, 0)
                bought = np.full(self.num_products, 0)
                started = np.full(self.num_products, 0)
                num_units = np.full(self.num_products, 0)
                appearance = 0
                for inter in interactions:
                    print(inter.getFeaturesValue())
                    if inter.getFeaturesValue()[0] == i and inter.getFeaturesValue()[1] == j:
                        appearance += 1
                        visits = np.add(visits, inter.linearizeVisits())
                        bought = np.add(bought, inter.linearizeBought())
                        started = np.add(started, inter.linearizeStart())
                        num_units = np.add(num_units, inter.linearizeNumUnits())
                self.contexts_visits.append(visits)
                self.contexts_bought.append(bought)
                self.contexts_started.append(started)
                self.contexts_units.append(num_units)

        # Using visits/bought/started/units we can compute CR, alphas and units mean for each context
        # and use it to compute the context expected value
        # Also summing on started we can get the exact number of user in that context, to compute context probability

        return


# Main for testing purposes
files = ['./Configs/config1.json', './Configs/config3.json', './Configs/configuration4.json',
         './Configs/configuration5.json']
env = []
config_margins = []
optimal_arms = []
conv_rates = []
prod_lists = []
click_probs = []
lambdas = []
alphas = []
units_means = []
clairvoyant_opt_rew = []
actual_unit_mean = []

n_loops = 1

for i in range(0, len(files)):
    env.append(Environment(config_path=files[i]))
    config = mergeUserClasses([files[i]], False)[0]
    config_margins.append(config["marginsPerPrice"])
    optimal_arms.append(config["optimalConfig"])
    conv_rates.append(config["conversionRateLevels"])
    prod_lists.append(config["productList"])
    print("ProdList={}, Alphas={}, ConvRates={}".format(len(config["productList"]), len(config["alphas"]),
                                                        len(config["conversionRateLevels"])))
    click_probs.append(config["click_prob"])
    lambdas.append(config['lambda_p'])
    alphas.append(config["alphas"])
    clairvoyant_opt_rew.append(config["optimalMargin"])
    units_means.append(config["units_mean"])
    actual_unit_mean.append(config["actual_units_mean"])

env[0].setPriceLevels([0, 0, 0, 0, 0])
ret = env[0].round()
contextGen = ContextGeneration()
contextGen.computePartition(interactions=ret, optimal_arm=[0, 0, 0, 0, 0])
