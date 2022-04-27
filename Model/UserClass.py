from asyncio.windows_events import NULL
import numpy as np
from Model.InteractionNode import InteractionNode
import math

class UserClass:
    """
    This class simulate the behaviour of a specific class of users. The users belonging to the class could have different features value.

    Attributes
    ----------
    conversionRate : matrix
        A matrix of size #Products (rows) by #Prices (columns), e.g. 2 prods and 3 prices -> [[.3,.4,.1],[.43,.98,.02]]
    clickProbability : GraphProbabilities
        Object of type GraphProbabilities that states the probability of clicking a product given that it has been seen, i.e. P(click i | seen i)
    alphas : vector
        Vector of floats that describe the probability of starting from a specific product, length must be equal to #Products
    productList : vector(Product) 
        Vector of objects of type products
    units_gamma_shape : int
        Shape parameter of numpy gamma distribution used to model the number of units bought
    units_gamma_scale : float
        Scale parameter of numpy gamma distribution used to model the number of units bought
    n_user_mean : int
        Mean of a gaussian distribution for daily users 
    n_user_variance : int
        Variance of a gaussian distribution for daily users 
    Lambda : float
        Probability to see the second secondary product given that the first one is seen, i.e. P(see sec(i,2) | see sec(i,1)) = Lambda
    features_generator : vector[dict["name","probability"]]
        Binary features generator in which probability is P(bin_feat=1), it's an array of dict like this: [{"name":"Tall >180cm","probability":0.2}]
    debug : bool
        Print additional information about the execution

    Methods
    -------
    setCurrentPrice(currentPrice)
        Set the currently used price levels for the following interactions
    generateEpisode()
        Generate a session/interaction for a new user. It returns a InteractionNode object
    """

    #def __init__(self, id, conversionRate, productList, clickProbability, alphas, n_bought_mean, n_bought_variance, n_user_mean, n_user_variance, Lambda):
    def __init__(self, conversionRate=[], clickProbability=NULL, alphas=[], units_gamma_shape=3, units_gamma_scale=1, 
                n_user_mean=15, n_user_variance=4, productList=[], Lambda=0.4, debug=True, features_generator=[]):
        """
        Parameters
        ----------
        conversionRate : matrix
            A matrix of size #Products (rows) by #Prices (columns), e.g. 2 prods and 3 prices -> [[.3,.4,.1],[.43,.98,.02]]
        
        clickProbability : GraphProbabilities
            Object of type GraphProbabilities that states the probability of clicking a product given that it has been seen, i.e. P(click i | seen i)
        
        alphas : vector
            Vector of floats that describe the probability of starting from a specific product, length must be equal to #Products
        
        productList : vector(Product) 
            Vector of objects of type products
        
        units_gamma_shape : int
            Shape parameter of numpy gamma distribution used to model the number of units bought
        
        units_gamma_scale : float
            Scale parameter of numpy gamma distribution used to model the number of units bought
        
        n_user_mean : int
            Mean of a gaussian distribution for daily users 
        
        n_user_variance : int
            Variance of a gaussian distribution for daily users 
        
        Lambda : float
            Probability to see the second secondary product given that the first one is seen, i.e. P(see sec(i,2) | see sec(i,1)) = Lambda
        
        features_generator : vector[dict["name","probability"]]
            Binary features generator in which probability is P(bin_feat=1), it's an array of dict like this: [{"name":"Tall >180cm","probability":0.2}]
        
        debug : bool
            Print additional information about the execution
        """

        self.conversionRate = conversionRate

        self.clickProbability = clickProbability
        assert clickProbability != NULL

        self.alphas = alphas
        assert np.array(alphas).sum() == 1
        assert len(conversionRate) == len(alphas) and len(alphas) == len(productList)
        # From alpha_i = [.3, .4, .3] generate [.3, .7, 1]
        self.product_alphas_intervals = np.full((len(self.alphas)), 0, dtype=float)
        cumulative = 0
        for i in range(0, len(self.alphas)):
            cumulative = cumulative + self.alphas[i]
            self.product_alphas_intervals[i] = cumulative
        self.product_alphas_intervals = np.array(self.product_alphas_intervals)    
        if debug: print(self.product_alphas_intervals)

        self.n_user = (n_user_mean, n_user_variance)

        self.productList = productList

        self.Lambda = Lambda
        assert 0 <= self.Lambda <= 1
        self.currentPrice = []
        self.units_gamma_shape = units_gamma_shape
        self.units_gamma_scale = units_gamma_scale
        self.debug = debug
        
        self.features_prob = []
        self.features_names = []
        for i in range(0,len(features_generator)):
            self.features_prob.append(features_generator[i]["probability"])
            self.features_names.append(features_generator[i]["name"])
        self.features_prob = np.array(self.features_prob)

    def generateEpisode(self):
        # For a user simulate the interaction with the website, returning clicks and currentProduct bought
        rnd = np.random.uniform(low=0.0, high=1.0, size=None)
        currentProduct = 0

        # Get the index for which the value exceeds the generated random number
        # [.3, .4, .3] --> [.3, .7, 1]; If rnd=0.45 then currentProduct = 1
        currentProduct = np.argmax(self.product_alphas_intervals > rnd)    
        assert 0 <= currentProduct < len(self.alphas)

        # Initialize the history of products to be visited
        history = np.full((len(self.alphas)), 1, dtype=int)

        userInteractions = self.generateProductInteraction(currentProduct, history)
        userInteractions.setFeatures(self.features_names, self.generateFeature())
        return userInteractions

    def setCurrentPrice(self, currentPrice):
        self.currentPrice = currentPrice

    def generateFeature(self):
        gen_arr = np.random.rand(len(self.features_prob))
        return gen_arr < self.features_prob

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
            firstSlot = self.productList[currentProduct].getSecondaryProduct(0)
            if self.debug: print('sec1: ', firstSlot)
            secondSlot = self.productList[currentProduct].getSecondaryProduct(1)
            if self.debug: print('sec2: ', secondSlot)

            # variables 'clickProbSec1' and 'clickProbSec2' are the click probabilities associated to the two
            # secondary products 'sec1' and 'sec2 NB: the click of the secondary product in the second slot (sec2)
            # has to be multiplied by the factor 'Lambda'
            clickProbSec1 = self.clickProbability.getEdgeProbability(currentProduct, firstSlot) * history[firstSlot]
            clickProbSec2 = self.clickProbability.getEdgeProbability(currentProduct, secondSlot) * self.Lambda * history[secondSlot]

            # variable 'sec1Bought' is the outcome of the binomial (number of successful trials): if the product is
            # bought rnd will be equal to 1 the same applies for variable 'sec2Bought'
            sec1Bought = np.random.binomial(1, clickProbSec1)
            if self.debug: print('sec1Bought: ', sec1Bought)
            if sec1Bought == 1:
                following.append(self.generateProductInteraction(firstSlot, history))

            sec2Bought = np.random.binomial(1, clickProbSec2)
            if self.debug: print('sec2Bought: ', sec2Bought)
            if sec2Bought == 1:
                following.append(self.generateProductInteraction(secondSlot, history))



        # At the end of the interaction between the user and the current product an INTERACTION NODE is generated to
        # keep track of the user history
        interactionNode = InteractionNode(product=currentProduct, price=self.currentPrice[currentProduct],
                                          firstSlot=firstSlot, secondSlot=secondSlot, sec1Bought=sec1Bought,
                                          sec2Bought=sec2Bought, bought=bought, units=units, following=following)
        return interactionNode