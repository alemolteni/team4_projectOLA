import numpy as np


class UserClass:
    def __init__(self, id, conversionRate, graphConnections, graphProbability, alphas, n_bought_mean, n_bought_variance, n_user_mean, n_user_variance, Lambda):
        self.id = id
        self.conversionRate = conversionRate
        self.graphProbability = graphProbability
        self.alphas = alphas
        self.n_bought = [n_bought_mean, n_bought_variance]
        self.n_user = [n_user_mean, n_user_variance]
        self.graphConnections = graphConnections
        self.Lambda = Lambda
        self.currentPrice = []

    def generateEpisode(self):
        #For a user simulate the interaction with the website, returning clicks and product bought
        rnd = np.random()

        product = 0
        cumulative = 0
        for i in range(0, len(self.alphas)):
            cumulative = cumulative + self.alphas[i]
            if rnd <= cumulative:
                product = i
                break;
        history = [for i in range(0,len(self.alphas)) 1]
        return self.generateProductInteraction(product, hi)

    def setCurrentPrice(self, currentPrice):
        self.currentPrice = currentPrice

    def generateProductInteraction(self, product, history):

        buyingProb = self.conversionRate[product][self.currentPrice[product]]
        bought = np.random.binomial(1, buyingProb)

        units = 0
        following = []
        history[product] = 0
        if bought == 1:
            units = np.random.exponential(self.n_bought[1], 0)
            sec1 = graph.getSecondary(1)
            sec2 = graph.getSecondary(2)
            probSec1 = graph.getSecondaryProbability(1) * history[sec1]
            probSec2 = graph.getSecondaryProbability(2) * self.Lambda * history[sec2]
            rnd = np.random.binomial(1, probSec1)
            if rnd == 1:
                result = self.generateProductInteraction(graph.getSecondary(1), history)
                following.append(result)
            rnd = np.random.binomial(1, probSec2)
            if rnd == 1:
                following.append(self.generateProductInteraction(graph.getSecondary(2), history))

        interactionNode = new InteractionNode(product=product, currentPrice=self.currentPrice[product], bought=bought,
                                              units=units,following=following)
        return interactionNode