from Environment import *
from Learner.TS_Alphas import TS_Alphas
from Model.UserClass import *
from Model.Product import *
from Model.constants import *
from Model.GraphProbabilities import *
from Model.Evaluator.GraphEvaluator import *

from Learner.GreedyLearner import *
from Learner.TS_CR import *
from Learner.TS_GW import *
from Learner.UCB_CR import *
from Learner.BruteForce import *
import numpy as np
import matplotlib.pyplot as plt


def totalMarginPerNode(envReturn, marginsPerPrice, pulledArm):
    TotMargin = 0

    for e in envReturn:
        # print("Product: ", e.product,
        #     "Price: ", pulledArm[e.product],
        #    "Units: ", e.units,
        #   "Margin: ", marginsPerPrice[e.product][pulledArm[e.product]] * e.units)
        if e.bought:
            TotMargin += marginsPerPrice[e.product][pulledArm[e.product]] * e.units
    return TotMargin


def testTS_CR():
    f = open('Configs/config1.json')
    config = json.load(f)
    margins = config["margins"]
    user_config = config["classes"][0]
    conv_rates = user_config["conversionRates"]
    alphas = user_config["alphas"]
    click_prob = user_config["clickProbability"]
    secondary_prod = [Product(int(key), user_config["secondary"][key]) for key in user_config["secondary"]]
    l = user_config["lambda"]

    for i in range(0, 5):
        print('{} - {} - {} - {}'.format(margins[i][0] * conv_rates[i][0], margins[i][1] * conv_rates[i][1],
                                         margins[i][2] * conv_rates[i][2], margins[i][3] * conv_rates[i][3]))

    print('\n')

    tsLearner = TS_CR(margins=margins, alphas=alphas, secondary_prod=secondary_prod, click_prob=click_prob, l=l)
    n_experiments = 100
    environment = Environment(config_path="Configs/config1.json")

    for i in range(0, n_experiments):
        pulledArm = tsLearner.pull_arm()
        environment.setPriceLevels(pulledArm)
        interactions = environment.round()
        tsLearner.update(interactions, pulledArm)

    print("Pulled arm {} at time {}:".format(pulledArm, i))
    print("Optimal configuration in theory: [2, 2, 0, 3, 2]")

    print(tsLearner.estimated_conversion_rates)

def testTS_GW():
    f = open('Configs/config1.json')
    config = json.load(f)
    margins = config["margins"]
    user_config = config["classes"][0]
    conv_rates = user_config["conversionRates"]
    alphas = user_config["alphas"]
    click_prob = user_config["clickProbability"]
    secondary_prod = [Product(int(key), user_config["secondary"][key]) for key in user_config["secondary"]]
    l = user_config["lambda"]

    """
    for i in range(0, 5):
        print('{} - {} - {} - {}'.format(margins[i][0] * conv_rates[i][0], margins[i][1] * conv_rates[i][1],
                                         margins[i][2] * conv_rates[i][2], margins[i][3] * conv_rates[i][3]))
                                         """

    print('\n')

    tsLearner = TS_GW(margins=margins, alphas=alphas, secondary_prod=secondary_prod, conversion_rates=conv_rates, l=l)
    n_experiments = 100
    environment = Environment(config_path="Configs/config1.json")

    print(tsLearner.estimated_click_prob)

    for i in range(0, n_experiments):
        pulledArm = tsLearner.pull_arm()
        environment.setPriceLevels(pulledArm)
        interactions = environment.round()
        tsLearner.update(interactions, pulledArm)

    print("Pulled arm {} at time {}:".format(pulledArm, i))
    print("Optimal configuration in theory: [2, 2, 0, 3, 2]")

    print(tsLearner.estimated_click_prob)


def testUCB_CR():
    env = Environment(config_path="Configs/config1.json")
    margins = [
        [1, 2, 10, 16],
        [10, 20, 35, 40],
        [12, 15, 18, 21],
        [3, 8, 15, 21],
        [8, 10, 17, 26]
    ]
    conv_rates = [[0.7, 0.7, 0.4, 0.2],
                  [0.9, 0.8, 0.6, 0.2],
                  [0.9, 0.7, 0.5, 0.3],
                  [0.8, 0.7, 0.4, 0.3],
                  [0.9, 0.65, 0.45, 0.2]]

    clickProb = [
      [0, 0.5, 0.6, 0, 0],
      [0, 0, 0.4, 0, 0.2],
      [0, 0, 0, 0.8, 0.7],
      [0.6, 0, 0, 0, 0.5],
      [0, 0.9, 0, 0.3, 0]
    ]

    alphas = [0.3, 0.25, 0.15, 0.15, 0.15]

    secondary = {0: [1, 2], 1: [2, 4], 2: [3, 4], 3: [4, 0], 4: [1, 3]} #Changed secondary sintax to avoid strings

    learner = UCB_CR(margins=margins, clickProbability=clickProb, alphas=alphas, secondary=secondary, Lambda=0.7, debug=True)
    #learner = UCB(margins=margins)
    n_experiments = 100

    for t in range(1, n_experiments):
        conf = learner.pull_arm()
        print(conf)
        env.setPriceLevels(conf)
        rew = env.round()
        learner.update(rew)

    print("Pulled arm {} at time {}:".format(conf, t))

    return


def testGreedy():
    environment = Environment(config_path="Configs/config1.json")
    gLearner = GreedyLearner(debug=True)

    marginsPerPrice = [
        [1, 2, 10, 16],
        [10, 20, 35, 40],
        [12, 15, 18, 21],
        [3, 8, 15, 21],
        [8, 10, 17, 26]
    ]

    n_experiments = 100

    optimal_arm = []

    allMargin = np.array([])

    for i in range(0, 1030):
        pulledArm = gLearner.pull_arm()
        print(pulledArm)
        environment.setPriceLevels(pulledArm)
        envReturn = environment.round()

        price_configuration_margin = 0
        for j in range(0, n_experiments):
            price_configuration_margin += totalMarginPerNode(envReturn, marginsPerPrice, pulledArm)
            envReturn = environment.round()

        margin = price_configuration_margin / n_experiments
        print(margin)
        allMargin = np.append(allMargin, margin)
        gLearner.update(margin)

    print("Optima", gLearner.get_optima())
    print("Optima margin", gLearner.get_optima_margin())
    x = np.linspace(0, 1030, 1030)

    fig, ax = plt.subplots()
    ax.plot(x, allMargin)
    plt.show()
    # if np.array_equal(optimal_arm, pulledArm):
    # break

    # optimal_arm = pulledArm

    # print(optimal_arm)



def arrays_mean(arrays, weights):
        tot_weights = sum(weights)
        if len(arrays) == 0:
            return []

        ret = arrays[0]
        for i in range(0, len(ret)):
            ret[i] = ret[i] * weights[0]

        for i in range(1, len(arrays)):
            for j in range(0, len(arrays[0])):
                ret[j] += arrays[i][j] * weights[i]

        for i in range(0, len(ret)):
            ret[i] = ret[i] / (len(arrays) * tot_weights)

        return ret

def matrices_mean(matrices, weights):
        tot_weights = sum(weights)
        if len(matrices) == 0:
            return []

        ret = matrices[0]
        for i in range(0, len(ret)):
            for j in range(0, len(ret[0])):
                ret[i][j] = ret[i][j] * weights[0]

        for i in range(1, len(matrices)):
            for j in range(0, len(matrices[0])):
                for k in range(0, len(matrices[0][0])):
                    ret[j][k] += matrices[i][j][k] * weights[i]

        for i in range(0, len(ret)):
            for j in range(0, len(ret[0])):
                ret[i][j] = ret[i][j] / (len(matrices) * tot_weights)

        return ret

def load_mean_config(file_path):
    f = open(file_path)
    config = json.load(f)
    f.close()
    ucs = config["classes"]
    margins = config["margins"]
    optimal_config = config["optimalConfig"]
    productList = [Product(int(key), ucs[0]["secondary"][key]) for key in
                    ucs[0]["secondary"]]  # same for each class

    convRates = []
    click_prob = []
    lambda_p = 0
    alphas = []
    units_mean = []
    users_mean = []
    users_variance = []

    for uc in ucs:
        # medie
        convRates.append(uc["conversionRates"])
        click_prob.append(uc["clickProbability"])
        lambda_p += uc["lambda"]
        alphas.append(uc["alphas"])
        units_mean.append(uc["unitsShape"])
        users_mean.append(uc["usersMean"])
        users_variance.append(uc["usersVariance"])

    lambda_p = lambda_p / len(ucs)
    alphas = arrays_mean(alphas, users_mean)
    units_mean = arrays_mean(units_mean, users_mean)
    click_prob = matrices_mean(click_prob, users_mean)
    convRates = matrices_mean(convRates, users_mean)

    num_prices = len(convRates[0])
    num_prods = len(alphas)

    pulledArm = [0, 3, 0, 2, 0]  # [3, 2, 0, 3, 2]
    configuration = {"productList": productList, "click_prob": click_prob, "lambda_p": lambda_p, "alphas": alphas,
                     "units_mean": units_mean,
                      "num_prices": num_prices, "num_prods": num_prods, "optimal_config": optimal_config,
                      "pulledArm": pulledArm,
                     "margins": margins, "convRates": convRates, "usersMean": users_mean,
                      "usersVariance": users_variance}
    return configuration

    def total_margin_per_node(envReturn, marginsPerPrice, pulledArm):
        TotMargin = 0
        for e in envReturn["episodes"]:
            if e.bought:
                TotMargin += marginsPerPrice[e.product][pulledArm[e.product]] * e.units
        return TotMargin

def testTS_Alphas():
    files = ['./Configs/config1.json', './Configs/config3.json']
    # files = ['./Configs/config1.json', './Configs/Config2.json']
    env = []
    tsLearners = []
    config_margins = []
    optimal_arms = []
    conv_rates = []
    prod_lists = []
    click_probs = []
    lambdas = []
    alphas = []
    units_means = []
    n_experiments = 100
    n_loops = 1

    for i in range(0, len(files)):
        env.append(Environment(config_path=files[i]))
        config = load_mean_config(files[i])
        l = config["lambda_p"]
        tsLearners.append(TS_Alphas(margins=config["margins"], secondary_prod=config["productList"],
                                    click_prob=config["click_prob"], l=l))
        config_margins.append(config["margins"])
        optimal_arms.append(config["optimal_config"])
        conv_rates.append(config["convRates"])
        prod_lists.append(config["productList"])
        click_probs.append(config["click_prob"])
        lambdas.append(config['lambda_p'])
        alphas.append(config["alphas"])
        units_means.append(config["units_mean"])

    tot_ts_learner_margins = []
    tot_optimal_margins = []

    for i in range(0, len(env)):
        ts_learner_margins = np.array([])
        optimal_margins = np.array([])
        for j in range(0, n_experiments):
            ts_single_margin = 0
            opt_single_margin = 0
            armMargins = []
            armConvRates = []

            # compute the margin for the TS
            pulledArm = tsLearners[i].pull_arm()
            for k in range(0, len(pulledArm)):
                armMargins.append(config_margins[i][k][pulledArm[k]])
                armConvRates.append(conv_rates[i][k][pulledArm[k]])

            graphEval = GraphEvaluator(products_list=prod_lists[i], click_prob_matrix=click_probs[i],
                                       lambda_prob=lambdas[i],
                                       alphas=alphas[i], conversion_rates=armConvRates, margins=armMargins,
                                       units_mean=units_means[i], verbose=False)

            env[i].setPriceLevels(pulledArm)
            for k in range(0, n_loops):
                ts_interactions = env[i].round()
                ts_single_margin += graphEval.computeMargin()
                # ts_single_margin += total_margin_per_node(ts_interactions, config_margins[i], pulledArm)
            ts_single_margin /= n_loops
            tsLearners[i].update(ts_interactions, pulledArm)

            # compute the margin for the optimal
            armConvRates = []
            armMargins = []
            pulledArm = optimal_arms[i]
            for k in range(0, len(pulledArm)):
                armMargins.append(config_margins[i][k][pulledArm[k]])
                armConvRates.append(conv_rates[i][k][pulledArm[k]])

            graphEval = GraphEvaluator(products_list=prod_lists[i], click_prob_matrix=click_probs[i],
                                       lambda_prob=lambdas[i],
                                       alphas=alphas[i], conversion_rates=armConvRates, margins=armMargins,
                                       units_mean=units_means[i], verbose=False)

            env[i].setPriceLevels(pulledArm)
            for k in range(0, n_loops):
                opt_interactions = env[i].round()
                opt_single_margin += graphEval.computeMargin()
                # opt_single_margin += total_margin_per_node(opt_interactions, config_margins[i], pulledArm)
            opt_single_margin /= n_loops

            # add the margins
            ts_learner_margins = np.append(ts_learner_margins, ts_single_margin)
            optimal_margins = np.append(optimal_margins, opt_single_margin)

        tot_ts_learner_margins.append(ts_learner_margins)
        tot_optimal_margins.append(optimal_margins)
        print("Configuration file ", i, ":\nOptimal arm found:\n\t", tsLearners[i].pull_arm(),
              "\nOptimal theoretical arm:\n\t", optimal_arms[i])
        print("Final alphas: \n", tsLearners[i].alphas)



testTS_GW()
