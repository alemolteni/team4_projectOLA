import numpy as np
from Environment import *

class GreedyLearner:

    def __init__(self, prices):
        self.configuration_margin = [0, 0, 0, 0, 0] #margini da valutare, ognuno per una delle 5 configurazioni di prezzo
        self.t = -1 #per provare la configurazione tutti 0
        self.optimal_configuration = [0, 0, 0, 0, 0] #configurazione di prezzo ottimale
        self.optimal_configuration_margin = 0 #margine ottimale della config sopra
        self.prices = prices
        self.isOptima = False

    def pull_arm(self):
        if self.t == -1 or self.isOptima:
            return self.optimal_configuration

        try_configuration = self.optimal_configuration.copy()
        if try_configuration[self.t] < 4:
            try_configuration[self.t] += 1

        return try_configuration

    def update(self, margin):
        if self.t == -1:
            self.optimal_configuration_margin = margin
            self.t += 1
            return

        self.configuration_margin[self.t] = margin
        self.t += 1
        #caso speciale se t = 4 -> aggiornare con la migliore config e abbiamo una nuova configurazione migliore
        if self.t == 5:
            if np.amax(self.configuration_margin) >= self.optimal_configuration_margin:
                result = np.where(self.configuration_margin == np.amax(self.configuration_margin))
                self.optimal_configuration[result] += 1
                self.optimal_configuration_margin = margin
            else: self.isOptima = True
            self.t = 0
            self.configuration_margin = [0, 0, 0, 0, 0]

