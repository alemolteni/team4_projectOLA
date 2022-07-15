import numpy as np


class BruteForce:

    def __init__(self, num_products=5, num_prices=4, debug=False):
        self.current_config = [0 for i in range(0,num_products)]
        self.t = -1  # per provare la configurazione tutti 0
        self.optimal_configuration = [0, 0, 0, 0, 0]  # configurazione di prezzo ottimale
        self.optimal_configuration_margin = 0  # margine ottimale della config sopra
        self.isOptima = False
        self.numberOfComparison = 5
        self.debug = debug
        self.product_index = 0
        self.num_products = num_products
        self.num_prices = num_prices
        self.previous_max = 0

    def numberToBase(self, n, b):
        if n == 0:
            return [0]
        digits = []
        while n:
            digits.append(int(n % b))
            n //= b
        return digits[::-1]
    
    def pull_arm(self):
        if self.t == self.num_prices**self.num_products - 1:
            return self.optimal_configuration
        self.t  += 1

        partial_config = self.numberToBase(self.t,self.num_prices)
        remaining = [0 for i in range(0,self.num_products-len(partial_config))]
        self.current_config = remaining + partial_config

        return self.current_config

    def update(self, overallMargin):
        if self.optimal_configuration_margin < overallMargin:
            self.previous_max = self.optimal_configuration_margin 
            self.optimal_configuration = self.current_config.copy()
            self.optimal_configuration_margin = overallMargin

    def get_optima(self):
        return self.optimal_configuration

    def get_optima_margin(self):
        return self.optimal_configuration_margin

    def get_delta_min(self):
        return self.optimal_configuration_margin - self.previous_max