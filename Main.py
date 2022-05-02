from Environment import *
from Model.UserClass import *
from Model.Product import *
from Model.constants import *
from Model.GraphProbabilities import *

from Learner.GreedyLearner import *


def totalMarginPerNode(envReturn, marginsPerPrice, pulledArm):
    TotMargin = 0

    for e in envReturn:
        #print("Product: ", e.product,
         #     "Price: ", pulledArm[e.product],
          #    "Units: ", e.units,
           #   "Margin: ", marginsPerPrice[e.product][pulledArm[e.product]] * e.units)
        if e.bought:
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
alphas = [0.2, 0.2, 0.2, 0.2, 0.2]

n_bought_gamma_shape = 2
n_bought_gamma_scale = 0.5
n_user_mean = 10
n_user_variance = 1

productList = [Product(int(key), SECONDARY_PRODUCTS[key]) for key in SECONDARY_PRODUCTS]

Lambda = 0.8

userClass = UserClass(conversionRate=conversionRate, clickProbability=clickProbability, debug=False, alphas=alphas,
                      Lambda=Lambda, n_user_mean=n_user_mean, n_user_variance=n_user_variance, productList=productList,
                      units_gamma_shape = n_bought_gamma_shape, units_gamma_scale = n_bought_gamma_scale,
                      features_generator=[{"name": "Over 18", "probability": 0.6},
                                          {"name": "Male", "probability": 0.9}])
environment = Environment([userClass])

gLearner = GreedyLearner(debug=True)

marginsPerPrice = [[1, 10, 1, 1],  # Product 1
                   [1, -1, 0, 0],  # Product 2
                   [1, 10, 1, 1],  # Product 3
                   [1, 10, 50, 1],  # Product 4
                   [1, 10, 100, 1]  # Product 5
                   ]

n_experiments = 100
optimal_arm = []

for i in range(0, 100):
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