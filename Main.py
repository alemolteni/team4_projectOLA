from Environment import *
from Model.UserClass import *
from Model.Product import *
from Model.constants import *
from Model.GraphProbabilities import *

from Learner.GreedyLearner import *
from Learner.TS_CR import *
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
    margins = [[1, 2, 10, 16],
               [10, 20, 35, 40],
               [12, 15, 18, 21],
               [3, 8, 15, 21],
               [8, 10, 17, 26]]
    conv_rates = [[0.7, 0.7, 0.4, 0.2],
                  [0.9, 0.8, 0.6, 0.2],
                  [0.9, 0.7, 0.5, 0.3],
                  [0.8, 0.7, 0.4, 0.3],
                  [0.9, 0.65, 0.45, 0.2]]

    for i in range(0, 5):
        print('{} - {} - {} - {}'.format(margins[i][0] * conv_rates[i][0], margins[i][1] * conv_rates[i][1],
                                         margins[i][2] * conv_rates[i][2], margins[i][3] * conv_rates[i][3]))

    print('\n')
    tsLearner = TS_CR(margins=margins)
    n_experiments = 100
    environment = Environment(config_path="Configs/config1.json")

    for i in range(0, n_experiments):
        pulledArm = tsLearner.pull_arm()
        environment.setPriceLevels(pulledArm)
        interactions = environment.round()
        tsLearner.update(interactions, pulledArm)

    print("Pulled arm {} at time {}:".format(pulledArm, i))
    print("Optimal configuration in theory: [2, 2, 0, 3, 2]")
    return;


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

    for i in range(0, 5):
        print('{} - {} - {} - {}'.format(margins[i][0] * conv_rates[i][0], margins[i][1] * conv_rates[i][1],
                                         margins[i][2] * conv_rates[i][2], margins[i][3] * conv_rates[i][3]))

    learner = UCB_CR(margins=margins)
    n_experiments = 100

    for t in range(1, n_experiments):
        conf = learner.pull_arm()
        print(conf)
        env.setPriceLevels(conf)
        rew = env.round()
        learner.update(rew)

    print("Pulled arm {} at time {}:".format(conf, t))
    print("Optimal configuration in theory: [2, 2, 0, 3, 2]")
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

testUCB_CR()
