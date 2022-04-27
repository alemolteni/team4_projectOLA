

class InteractionNode:
    """
    An 'interactionNode' is generated for each primary product that the user interacts with. At the end of the 'user
    episode' a list of interaction nodes is recursively generated in order to keep track of the actions performed by
    the user on the site
    """

    def __init__(self, product, price, bought, units, following):
        """
        Each interaction node keeps track of:
        - product: id of the product type involved in the interaction;
        - price: price of the product;
        - bought: equals to '1' if the product has been bought, '0' otherwise;
        - units: units bought;
        - following: list of the previous interaction nodes
        """
        self.product = product
        self.price = price
        self.bought = bought
        self.units = units
        self.following = following
        self.featuresNames = []
        self.featuresValues = []

    def setFeatures(self, featuresNames, featuresValues):
        self.featuresNames = featuresNames
        self.featuresValues = featuresValues
        assert len(featuresNames) == len(featuresValues)

    def printInteractions(self):
        featuresString = ""
        for i in range(0,len(self.featuresNames)):
            if i > 0:
               featuresString = featuresString + ", "
            featuresString = featuresString + self.featuresNames[i] + "=" + str(self.featuresValues[i])
            
        print('\nUSER INTERACTIONS: {}'.format(featuresString))
        self.printNode(0)


    def printNode(self, nodeLevel):
        boughtYN = 'No'
        levelFormat = ""
        for i in range(0,nodeLevel):
            levelFormat = levelFormat + "    "
        if self.bought == 1:
            boughtYN = 'Yes'
        print("{}╚══ Product={}, PriceLevel={}, Bought={}, #Units={}".format(levelFormat,self.product,self.price,boughtYN,self.units))

        for i in range(0,len(self.following)):
            self.following[i].printNode(nodeLevel + 1)
