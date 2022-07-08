from Learner.UCB.UCB_Step3 import UCB_Step3
import numpy as np


class UCB_SlidingWindow(UCB_Step3):

    def __init__(self, margins=np.ones((5, 4)), num_products=5, num_prices=4, debug=False, alphas=np.zeros(5),
                 clickProbability=np.zeros((5, 5)), secondary=None, Lambda=1, sliding_window_size=20, units_mean=None):
        # Can't use conversion rates and can't use alphas
        self.sw_size = sliding_window_size
        self.circular_buffer_visits = np.zeros((sliding_window_size, num_products))
        self.circular_buffer_bought = np.zeros((sliding_window_size, num_products))
        self.circular_buffer_configurations = np.zeros((sliding_window_size, num_products), dtype=int)
        self.headPoint = 0

        self.times_arms_bought = np.zeros((num_products, num_prices))

        self.initializeBuffer = True
        super(UCB_SlidingWindow, self).__init__(margins=margins, num_products=num_products, num_prices=num_prices,
                                                debug=debug,
                                                clickProbability=clickProbability, secondary=secondary, Lambda=Lambda,
                                                alphas=alphas, units_mean=units_mean)

    def update(self, interactions):
        # From daily interactions extract needed information, depending on step uncertainty:
        #   - Step 6: update belief over conversion rates using a sliding window

        visits = np.full(self.num_products, 0)
        bought = np.full(self.num_products, 0)

        for inter in interactions:
            visits = np.add(visits, inter.linearizeVisits())
            bought = np.add(bought, inter.linearizeBought())

        for i in range(0, self.num_products):
            self.times_arms_pulled[i][self.circular_buffer_configurations[self.headPoint][i]] -= \
                self.circular_buffer_visits[self.headPoint][i]
            self.times_arms_bought[i][self.circular_buffer_configurations[self.headPoint][i]] -= \
                self.circular_buffer_bought[self.headPoint][i]
            if self.times_arms_pulled[i][self.circular_buffer_configurations[self.headPoint][i]] < 0:
                self.times_arms_pulled[i][self.circular_buffer_configurations[self.headPoint][i]] = 0
            if self.times_arms_bought[i][self.circular_buffer_configurations[self.headPoint][i]] < 0:
                self.times_arms_bought[i][self.circular_buffer_configurations[self.headPoint][i]] = 0

        self.circular_buffer_configurations[self.headPoint] = self.configuration
        self.circular_buffer_bought[self.headPoint] = bought
        self.circular_buffer_visits[self.headPoint] = visits
        self.headPoint = (self.headPoint + 1) % self.sw_size

        for i in range(0, len(self.configuration)):
            self.times_arms_pulled[i][self.configuration[i]] += visits[i]
            self.times_arms_bought[i][self.configuration[i]] += bought[i]
            if self.times_arms_pulled[i][self.configuration[i]] > 0:
                self.conversion_rates[i][self.configuration[i]] = self.times_arms_bought[i][self.configuration[i]] / \
                                                                self.times_arms_pulled[i][self.configuration[i]]
            else:
                self.conversion_rates[i][self.configuration[i]] = 0.0

        if self.debug:
            # print(self.circular_buffer_bought)
            # print(self.circular_buffer_visits)
            print("Times arms pulled: ", self.times_arms_pulled)
            print("Conversion rates: ", self.conversion_rates)
        return
