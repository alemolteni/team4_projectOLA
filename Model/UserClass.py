import numpy as np
from Model.InteractionNode import InteractionNode


class UserClass:
    #def __init__(self, id, conversionRate, productList, clickProbability, alphas, n_bought_mean, n_bought_variance, n_user_mean, n_user_variance, Lambda):
    def __init__(self, **arguments):
        self.id = id
        self.conversionRate = arguments['conversionRate']
        self.clickProbability = arguments['clickProbability']
        self.alphas = arguments['alphas']

        self.n_bought = [arguments['n_bought_mean'], arguments['n_bought_variance']]

        self.n_user = [arguments['n_user_mean'], arguments['n_user_variance']]

        self.productList = arguments['productList']

        self.Lambda = arguments['Lambda']
        self.currentPrice = []

    def generateEpisode(self):
        # For a user simulate the interaction with the website, returning clicks and currentProduct bought
        rnd = np.random.uniform(low=0.0, high=1.0, size=None)
        print(rnd)

        currentProduct = 0
        cumulative = 0

        i = 0
        while i in range(0, len(self.alphas)):
            cumulative = cumulative + self.alphas[i]
            #print('rnd <= cumulative: ', rnd, ' <= ', cumulative)
            if rnd <= cumulative:
                currentProduct = i

                # loop exit condition:
                i = len(self.alphas)
            else:
                i += 1

        history = [1 for i in range(0, len(self.alphas))]

        userInteractions = self.generateProductInteraction(currentProduct, history)
        return userInteractions

    def setCurrentPrice(self, currentPrice):
        self.currentPrice = currentPrice


    def generateProductInteraction(self, currentProduct, history):

        # TODO: implement 'conversionRate'
        print('\ncurrentProduct: ', currentProduct)

        buyingProb = self.conversionRate[currentProduct-1][self.currentPrice[currentProduct-1]]
        bought = np.random.binomial(1, buyingProb)

        # variable 'units' keeps track of the units of product bought by the user, it's the result of an exponential
        # distribution <----- SHOULD BE UPDATED TO A GAMMA DISTRIBUTION
        units = 0
        following = []

        # List 'history' keeps track of the products that have been displayed as 'primary'
        #
        #   e.g.    Products 1 and 4 have been displayed as primary products in some previous iteration
        #           --> history = [0, 1, 1, 0, 1],
        #           the user then click on product 3, so product 3 is displayed as primary (generateProductInteraction()
        #           is summoned with product 3 as input).
        #           history is updated: --> history = [0, 1, 0, 0, 1]
        history[currentProduct] = 0

        # If the primary product displayed is bought that the click probability of the two linked secondary products
        # must be addressed
        if bought == 1:
            units = np.random.gamma(3, 1, None) # <-- TODO: UPDATE TO A GAMMA DISTRIBUTION
            units = round(units)
            print('units bought: ', units)

            # variables 'sec1' and 'sec2' are the two secondary products linked to the primary product that is being
            # displayed
            sec1 = self.productList[currentProduct-1].getSecondaryProduct(0)
            print('sec1: ', sec1)
            sec2 = self.productList[currentProduct-1].getSecondaryProduct(1)
            print('sec2: ', sec2)

            # variables 'clickProbSec1' and 'clickProbSec2' are the click probabilities associated to the two
            # secondary products 'sec1' and 'sec2 NB: the click of the secondary product in the second slot (sec2)
            # has to be multiplied by the factor 'Lambda'
            clickProbSec1 = self.clickProbability.getEdgeProbability(currentProduct, sec1) * history[sec1]
            clickProbSec2 = self.clickProbability.getEdgeProbability(currentProduct, sec2) * self.Lambda * history[sec2]

            # variable 'sec1Bought' is the outcome of the binomial (number of successful trials): if the product is
            # bought rnd will be equal to 1 the same applies for variable 'sec2Bought'
            sec1Bought = np.random.binomial(1, clickProbSec1)
            print('sec1Bought: ', sec1Bought)
            if sec1Bought == 1:
                result = self.generateProductInteraction(sec1, history)
                following.append(result)

            sec2Bought = np.random.binomial(1, clickProbSec2)
            print('sec2Bought: ', sec2Bought)
            if sec2Bought == 1:
                following.append(self.generateProductInteraction(sec2, history))



        # At the end of the interaction between the user and the current product an INTERACTION NODE is generated to
        # keep track of the user history
        interactionNode = InteractionNode(product=currentProduct, price=self.currentPrice[currentProduct-1],
                                          bought=bought, units=units, following=following)
        return interactionNode