from Environment import *
from Model.UserClass import *
from Model.Product import *
from Model.constants import *
from Model.GraphProbabilities import *

from Learner.GreedyLearner import *


def totalMarginPerNode(envReturn, marginsPerPrice, pulledArm):
    TotMargin = 0

    for e in envReturn:
        TotMargin += marginsPerPrice[e.product][pulledArm[e.product]] * e.units
    return TotMargin


conversionRate = [
            [1, 0.8, 0.6, 0.4],
            [1, 0.8, 0.6, 0.4],
            [1, 0.8, 0.6, 0.4],
            [1, 0.8, 0.6, 0.4],
            [1, 0.8, 0.6, 0.4]
        ]

clickProbability = GraphProbabilities(PROBABILITY_MATRIX)
alphas = [0.2, 0.1, 0.3, 0.1, 0.3]

n_bought_mean = 2
n_bought_variance = 1
n_user_mean = 10
n_user_variance = 1

productList = [Product(int(key), SECONDARY_PRODUCTS[key]) for key in SECONDARY_PRODUCTS]

Lambda = 0.8

userClass = UserClass(conversionRate=conversionRate, clickProbability=clickProbability, debug=False, alphas=alphas,
                      Lambda=Lambda, n_user_mean=n_user_mean, n_user_variance=n_user_variance, productList=productList,
                      features_generator=[{"name": "Over 18", "probability": 0.6},
                                          {"name": "Male", "probability": 0.9}])
environment = Environment([userClass])

gLearner = GreedyLearner()

marginsPerPrice = [[1, 2, 3, 4],  # Product 1
                   [1, 0, 0, 0],  # Product 2
                   [1, 2, 3, 4],  # Product 3
                   [1, 2, 3, 4],  # Product 4
                   [1, 2, 50, 100]  # Product 5
                   ]

n_experiments = 500
optimal_arm = []

for i in range(0, 300):
    pulledArm = gLearner.pull_arm()
    print(pulledArm)
    environment.setPriceLevels(pulledArm)
    envReturn = environment.round()

    price_configuration_margin = 0
    for j in range(0, n_experiments):
        price_configuration_margin += totalMarginPerNode(envReturn, marginsPerPrice, pulledArm)
        envReturn = environment.round()

    margin = price_configuration_margin / n_experiments
    gLearner.update(margin)

    #if np.array_equal(optimal_arm, pulledArm):
       #break

    #optimal_arm = pulledArm

#print(optimal_arm)