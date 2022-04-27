from Model import CLIcolors


class InteractionNode:
    """
    An 'interactionNode' is generated for each primary product that the user interacts with. At the end of the 'user
    episode' a list of interaction nodes is recursively generated in order to keep track of the actions performed by
    the user on the site
    """

    def __init__(self, product, price, bought, units, following, firstSlot, secondSlot, sec1Bought, sec2Bought):
        """
        Each interaction node keeps track of:
        - product: id of the product type involved in the interaction;
        - price: price of the product;
        - bought: equals to '1' if the product has been bought, '0' otherwise;
        - units: units bought;
        - following: list of the previous interaction nodes;
        - firstSlot: id of the secondary product in slot 1;
        - secondSlot: id of the secondary product in slot 2;
        - sec1Bought: equals to '1' if the secondary product in slot 1 has been bought, '0' otherwise;
        - sec2Bought: equals to '1' if the secondary product in slot 2 has been bought, '0' otherwise;
        """
        self.product = product
        self.price = price
        self.bought = bought
        self.units = units
        self.following = following
        self.featuresNames = []
        self.featuresValues = []
        self.firstSlot = firstSlot
        self.secondSlot = secondSlot
        self.sec1Bought = sec1Bought
        self.sec2Bought = sec2Bought

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
            
        print('\n{}USER INTERACTIONS: {}'.format(CLIcolors.bcolors.HEADER, featuresString))
        leftFiller = self.printNode(0)
        leftFiller = leftFiller + "    "
        self.interactionEndCLI(leftFiller)


    def printNode(self, nodeLevel):
        boughtYN = 'No'
        levelFormat = ""
        for i in range(0, nodeLevel):
            levelFormat = levelFormat + "    "
        if self.bought == 1:
            boughtYN = 'Yes'

        self.printBoxCLI(levelFormat, boughtYN)

        for i in range(0, len(self.following)):
            levelFormat = self.following[i].printNode(nodeLevel + 1)

        return levelFormat


    def printBoxCLI(self, levelFormat, boughtYN):

        boxWidth = 61 + len(levelFormat)
        fillChar = ' '
        print('{}┗━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓'.format(levelFormat))
        print('{}  ┃{} Product={}, PriceLevel={}, Bought={}, #Units={}'.format(levelFormat, CLIcolors.bcolors.BKGRD,
                                                                               self.product, self.price, boughtYN,
                                                                               self.units).ljust(boxWidth, fillChar),
              "{}┃".format(CLIcolors.bcolors.STD))
        print('{}  ┃{}   ■ FirstSlot={}, Bought={}'.format(levelFormat, CLIcolors.bcolors.BKGRD, self.firstSlot,
                                                           self.sec1Bought, CLIcolors.bcolors.STD).ljust(boxWidth,
                                                                                                         fillChar),
              '{}┃'.format(CLIcolors.bcolors.STD))
        print('{}  ┃{}   ■ SecondSlot={}, Bought={}'.format(levelFormat, CLIcolors.bcolors.BKGRD, self.secondSlot,
                                                            self.sec2Bought, CLIcolors.bcolors.STD).ljust(boxWidth,
                                                                                                          fillChar),
              '{}┃'.format(CLIcolors.bcolors.STD))
        print("{}  ┗━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛".format(levelFormat))


    @staticmethod
    def interactionEndCLI(leftFiller):
        print('{}┗━┳━━━━━━━━━━━━━━━━━━━┓'.format(leftFiller))
        print('{}  ┃{} INTERACTION ENDED {}┃'.format(leftFiller, CLIcolors.bcolors.REDBKGRD, CLIcolors.bcolors.STD))
        print("{}  ┗━━━━━━━━━━━━━━━━━━━┛".format(leftFiller))