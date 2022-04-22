import numpy as np


class UserClass:
    def __init__(self, id, conversionRate, graph, alphas, n_bought_mean, n_bought_variance, n_user_mean, n_user_variance):
        self.id = id
        self.conversionRate = conversionRate
        self.graph = graph
        self.alphas = alphas
        self.n_bought = [n_bought_mean, n_bought_variance]
        self.n_user = [n_user_mean, n_user_variance]

    def generateEpisode(self, Lambda):
        #For a user simulate the interaction with the website, returning clicks and product bought