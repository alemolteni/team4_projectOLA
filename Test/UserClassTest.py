import pprint
import sys
import unittest

print(sys.path)

from Model.GraphProbabilities import GraphProbabilities
from Model.Product import Product
from Model.UserClass import UserClass
from Model.constants import PROBABILITY_MATRIX, SECONDARY_PRODUCTS

class MyTestCase(unittest.TestCase):
    def test_UserEpisode(self):
        id = 0
        conversionRate = [
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1]
        ]

        clickProbability = GraphProbabilities(PROBABILITY_MATRIX)
        alphas = [0.2, 0.1, 0.3, 0.1, 0.3]

        n_bought_mean = 2
        n_bought_variance = 1
        n_user_mean = 10
        n_user_variance = 1

        productList = [Product(int(key), SECONDARY_PRODUCTS[key]) for key in SECONDARY_PRODUCTS]

        Lambda = 0.8

        userClass = UserClass(conversionRate=conversionRate, clickProbability=clickProbability, alphas=alphas,
                              Lambda=Lambda, n_user_mean=n_user_mean, n_user_variance=n_user_variance, productList=productList,
                              features_generator=[{"name":"Over 18","probability":0.6},{"name":"Male","probability":0.9}])

        # pp = pprint.PrettyPrinter(indent=4)
        # pp.pprint(userClass.__dict__)

        currentPrice = [0, 0, 0, 0, 0]
        userClass.setCurrentPrice(currentPrice)

        userInteractions = userClass.generateEpisode()
        userInteractions.printInteractions()





if __name__ == '__main__':
    
    unittest.main()
