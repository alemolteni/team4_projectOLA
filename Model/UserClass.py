import numpy as np
from Model.InteractionNode import InteractionNode


class UserClass:
    def __init__(self, id, conversionRate, productList, clickProbability, alphas, n_bought_mean, n_bought_variance, n_user_mean, n_user_variance, Lambda):
        self.id = id
        self.conversionRate = conversionRate
        self.clickProbability = clickProbability
        self.alphas = alphas
        self.n_bought = [n_bought_mean, n_bought_variance]
        self.n_user = [n_user_mean, n_user_variance]
        self.productList = productList
        self.Lambda = Lambda
        self.currentPrice = []

    def generateEpisode(self):
        # For a user simulate the interaction with the website, returning clicks and currentProduct bought
        rnd = np.random() # <------- TODO: MUST BE CHANGED TO A GAMMA DISTRIBUTION WITH Beta = 0.5

        currentProduct = 0
        cumulative = 0
        flag = True
        for i in range(0, len(self.alphas)) and flag:
            cumulative = cumulative + self.alphas[i]
            if rnd <= cumulative:
                currentProduct = i
                flag = False

        history = [1 for i in range(0,len(self.alphas))]
        return self.generateProductInteraction(currentProduct, history)

    def setCurrentPrice(self, currentPrice):
        self.currentPrice = currentPrice


    def generateProductInteraction(self, currentProduct, history):

        # TODO: implement 'conversionRate'
        buyingProb = self.conversionRate[currentProduct][self.currentPrice[currentProduct]]
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
            units = np.random.exponential(self.n_bought[1], 0) # <-- TODO: UPDATE TO A GAMMA DISTRIBUTION

            # variables 'sec1' and 'sec2' are the two secondary products linked to the primary product that is being
            # displayed
            sec1 = self.productList[currentProduct].getSecondaryProduct(0)
            sec2 = self.productList[currentProduct].getSecondaryProduct(1)

            # variables 'clickProbSec1' and 'clickProbSec2' are the click probabilities associated to the two
            # secondary products 'sec1' and 'sec2 NB: the click of the secondary product in the second slot (sec2)
            # has to be multiplied by the factor 'Lambda'
            clickProbSec1 = self.clickProbability.getSecondaryProbability(currentProduct, sec1) * history[sec1]
            clickProbSec2 = self.clickProbability.getSecondaryProbability(currentProduct, sec2) * self.Lambda * history[sec2]

            # variable 'sec1Bought' is the outcome of the binomial (number of successful trials): if the product is
            # bought rnd will be equal to 1 the same applies for variable 'sec2Bought'
            sec1Bought = np.random.binomial(1, clickProbSec1)
            if sec1Bought == 1:
                result = self.generateProductInteraction(self.productList.getSecondaryProduct(0), history)
                following.append(result)

            sec2Bought = np.random.binomial(1, clickProbSec2)
            if sec2Bought == 1:
                following.append(self.generateProductInteraction(self.productList.getSecondaryProduct(1), history))



        # At the end of the interaction between the user and the current product an INTERACTION NODE is generated to
        # keep track of the user history
        interactionNode = InteractionNode(product=currentProduct, price=self.currentPrice[currentProduct],
                                          bought=bought, units=units, following=following)

        return interactionNode