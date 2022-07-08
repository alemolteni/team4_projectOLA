import numpy as np


class GraphEvaluatorMC:
    def __init__(self, products_list, click_prob_matrix, lambda_prob, conversion_rates,
                 alphas, margins, units_mean):
        self.products_list = products_list
        self.click_prob = click_prob_matrix
        self.Lambda = lambda_prob
        self.conversion_rates = conversion_rates  # Of the chosen config (vector of length n_products)
        self.alphas = alphas
        self.margins = margins  # Of the chosen config (vector of length n_products)
        self.units_mean = units_mean

    def generateRandomLiveEdgeGraph(self, alpha):
        # Generate activation node
        rnd = np.random.uniform(low=0.0, high=1.0, size=None)
        starting = np.argmax(alpha > rnd)

        visitedNodes = [1] * len(self.alphas)
        boughtNodes = [0] * len(self.alphas)
        queue = [starting]
        while len(queue) > 0:
            p = queue.pop(0)
            cr = self.conversion_rates[p]
            if cr > 1:
                bought = 1
            else:
                bought = np.random.binomial(1, cr)
            if bought == 1:
                units = np.random.gamma(self.units_mean[p], 1, None)
                boughtNodes[p] += self.units_mean[p]
                firstSlot = self.products_list[p].getSecondaryProduct(0)
                secondSlot = self.products_list[p].getSecondaryProduct(1)
                clickSec1 = self.click_prob[p][firstSlot] * visitedNodes[firstSlot]
                clickSec2 = self.click_prob[p][secondSlot] * visitedNodes[secondSlot] * self.Lambda
                sec1Opened = np.random.binomial(1, clickSec1)
                sec2Opened = np.random.binomial(1, clickSec2)
                if sec1Opened == 1:
                    visitedNodes[firstSlot] = 0
                    queue.append(firstSlot)
                if sec2Opened == 1:
                    visitedNodes[secondSlot] = 0
                    queue.append(secondSlot)
        return boughtNodes

    # K chosen to have an error of less than 0.1 with a probability of 0.95 2273 - 19%
    def computeActivationProbability(self, k=2273):
        activations = [0] * len(self.alphas)
        alph = np.array(self.alphas) * 20
        alpha = np.cumsum(np.random.dirichlet(alph))
        for i in range(0, k):
            activations = np.add(activations, self.generateRandomLiveEdgeGraph(alpha))
        # print(activations)
        return activations / k

    def computeMargin(self):
        return np.multiply(self.computeActivationProbability(), self.margins).sum()


"""
import numpy as np
from Learner.BruteForce import *
from Learner.UCB.UCB_Step4 import *
from Model.ConfigurationParametersAverage import mergeUserClasses
from scipy.stats import gamma
from Model.Evaluator.GraphEvaluatorMC import *
import time
#files = ['./Configs/config1.json', './Configs/config2.json','./Configs/config3.json', './Configs/configDump.json', './Configs/configuration4.json', './Configs/configuration5.json']
files = ['./Configs/config2.json']
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
secondary = []
actual_unit_mean = []

for i in range(0, len(files)):
    env.append(Environment(config_path=files[i]))
    config = mergeUserClasses([files[i]], False)[0]
    config_margins.append(config["marginsPerPrice"])
    optimal_arms.append(config["optimalConfig"])
    conv_rates.append(config["conversionRateLevels"])
    prod_lists.append(config["productList"])
    print("ProdList={}, Alphas={}, ConvRates={}".format(len(config["productList"]),len(config["alphas"]),len(config["conversionRateLevels"])))
    click_probs.append(config["click_prob"])
    lambdas.append(config['lambda_p'])
    alphas.append(config["alphas"])
    clairvoyant_opt_rew.append(config["optimalMargin"])
    units_means.append(config["units_mean"])
    actual_unit_mean.append(config["actual_units_mean"])
# Check difference between Graph eval e environment margin
n_experiments = 100
n_user = [100, 70, 80, 95]
print(conv_rates[0])

for i in range(0, len(env)):
    maxMargin = -1
    optConf = [0, 0, 0, 0, 0]
    secondBestMargin = -1
    secondOptimal = [0, 0, 0, 0, 0]
    learner = BruteForce()
    rmse = 0

    for j in range(0, n_experiments):
        armMargins = []
        armConvRates = []
        pulledArm = learner.pull_arm()
        #print(pulledArm)
        for k in range(0, len(pulledArm)):
            armMargins.append(config_margins[i][k][pulledArm[k]])
            armConvRates.append(conv_rates[i][k][pulledArm[k]])

        graphEval = GraphEvaluatorMC(products_list=prod_lists[i], click_prob_matrix=click_probs[i], lambda_prob=lambdas[i],
                    alphas=alphas[i], conversion_rates=armConvRates, margins=armMargins, units_mean=actual_unit_mean[i])

        start = time.time()
        tempMargin = graphEval.computeMargin()
        #print(tempMargin, time.time() - start)

        env[i].setPriceLevels(pulledArm)
        mc_margin = 0
        start = time.time()
        for exp in range(0, 300):
            interactions = env[i].round()
            mc = 0
            for inter in interactions:
                mc = mc + inter.linearizeMargin(config_margins[i])
            mc_margin += mc / len(interactions)

        mc_margin = mc_margin / 300 # (100*n_user[i])

        rmse += (tempMargin - mc_margin)**2

        learner.update(tempMargin)
        if tempMargin > maxMargin:
            maxMargin = tempMargin
            optConf = pulledArm
        if tempMargin < maxMargin and tempMargin > secondBestMargin and pulledArm != secondOptimal:
            secondBestMargin = tempMargin
            secondOptimal = pulledArm

    config_name = files[i][files[i].rfind('/')-len(files[i])+1:]
    rmse = math.sqrt(rmse/n_experiments)
    print("RMSE: ", rmse)
    print("% error: ", (rmse/maxMargin)*100)
    print(config_name, "optimal is: ", optConf, maxMargin)
    print(config_name, "second optimal is: ", secondOptimal, secondBestMargin)
    
"""