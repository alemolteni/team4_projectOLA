import numpy as np
from Model.InteractionNode import InteractionNode
import math

class UserClass:
    #def __init__(self, id, conversionRate, productList, clickProbability, alphas, n_bought_mean, n_bought_variance, n_user_mean, n_user_variance, Lambda):
    def __init__(self, units_gamma_shape=3, units_gamma_scale=1, debug=True, **arguments):
        self.id = id
        self.conversionRate = arguments['conversionRate']
        self.clickProbability = arguments['clickProbability']

        self.alphas = arguments['alphas']
        # From alpha_i = [.3, .4, .3] generate [.3, .7, 1]
        self.product_alphas_intervals = np.full((len(self.alphas)), 0, dtype=float)
        cumulative = 0
        for i in range(0, len(self.alphas)):
            cumulative = cumulative + self.alphas[i]
            self.product_alphas_intervals[i] = cumulative
        self.product_alphas_intervals = np.array(self.product_alphas_intervals)    
        if debug: print(self.product_alphas_intervals)

        self.n_bought = [arguments['n_bought_mean'], arguments['n_bought_variance']]

        self.n_user = [arguments['n_user_mean'], arguments['n_user_variance']]

        self.productList = arguments['productList']

        self.Lambda = arguments['Lambda']
        assert self.Lambda >= 0 and self.Lambda <= 1
        self.currentPrice = []
        self.units_gamma_shape = units_gamma_shape
        self.units_gamma_scale = units_gamma_scale
        self.debug = debug

    def generateEpisode(self):
        # For a user simulate the interaction with the website, returning clicks and currentProduct bought
        rnd = np.random.uniform(low=0.0, high=1.0, size=None)
        currentProduct = 0

        # Get the index for which the value exceeds the generated random number
        # [.3, .4, .3] --> [.3, .7, 1]; If rnd=0.45 then currentProduct = 1
        currentProduct = np.argmax(self.product_alphas_intervals > rnd)    
        assert currentProduct >= 0 and currentProduct < len(self.alphas)   

        # Initialize the history of products to be visited
        history = np.full((len(self.alphas)), 1, dtype=int)

        userInteractions = self.generateProductInteraction(currentProduct, history)
        return userInteractions

    def setCurrentPrice(self, currentPrice):
        self.currentPrice = currentPrice


    def generateProductInteraction(self, currentProduct, history):
        # TODO: implement 'conversionRate'
        if self.debug: print('\ncurrentProduct: ', currentProduct)

        buyingProb = self.conversionRate[currentProduct][self.currentPrice[currentProduct]]
        bought = np.random.binomial(1, buyingProb)

        # variable 'units' keeps track of the units of product bought by the user, it's the result a gamma distribution
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
            units = np.random.gamma(self.units_gamma_shape, self.units_gamma_scale, None)
            units = math.ceil(units) # Ceil bc we weant at least one unit
            assert units > 0

            if self.debug: print('units bought: ', units)

            # variables 'sec1' and 'sec2' are the two secondary products linked to the primary product that is being
            # displayed
            sec1 = self.productList[currentProduct].getSecondaryProduct(0)
            if self.debug: print('sec1: ', sec1)
            sec2 = self.productList[currentProduct].getSecondaryProduct(1)
            if self.debug: print('sec2: ', sec2)

            # variables 'clickProbSec1' and 'clickProbSec2' are the click probabilities associated to the two
            # secondary products 'sec1' and 'sec2 NB: the click of the secondary product in the second slot (sec2)
            # has to be multiplied by the factor 'Lambda'
            clickProbSec1 = self.clickProbability.getEdgeProbability(currentProduct, sec1) * history[sec1]
            clickProbSec2 = self.clickProbability.getEdgeProbability(currentProduct, sec2) * self.Lambda * history[sec2]

            # variable 'sec1Bought' is the outcome of the binomial (number of successful trials): if the product is
            # bought rnd will be equal to 1 the same applies for variable 'sec2Bought'
            sec1Bought = np.random.binomial(1, clickProbSec1)
            if self.debug: print('sec1Bought: ', sec1Bought)
            if sec1Bought == 1:
                following.append(self.generateProductInteraction(sec1, history))

            sec2Bought = np.random.binomial(1, clickProbSec2)
            if self.debug: print('sec2Bought: ', sec2Bought)
            if sec2Bought == 1:
                following.append(self.generateProductInteraction(sec2, history))



        # At the end of the interaction between the user and the current product an INTERACTION NODE is generated to
        # keep track of the user history
        interactionNode = InteractionNode(product=currentProduct, price=self.currentPrice[currentProduct],
                                          bought=bought, units=units, following=following)
        return interactionNode