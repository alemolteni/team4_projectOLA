from Learner.UCB.UCB_Step3 import UCB_Step3
from Model.Evaluator.GraphEvaluator import GraphEvaluator
import numpy as np
import math


class UCB_Step5(UCB_Step3):
    def __init__(self, margins=np.ones((5, 4)), num_products=5, num_prices=4, debug=False,
                 alphas=np.zeros(5), conversion_rates=np.zeros((5, 4)), secondary=None, Lambda=1, units_mean=None):
        # Can't use click probability
        self.trial_click_first = np.zeros(num_products)
        self.trial_click_second = np.zeros(num_products)
        self.times_first_opened = np.zeros(num_products)
        self.times_second_opened = np.zeros(num_products)

        super(UCB_Step5, self).__init__(margins=margins, num_products=num_products, num_prices=num_prices, debug=debug,
                                        alphas=alphas, secondary=secondary, Lambda=Lambda,
                                        conversion_rates=conversion_rates, units_mean=units_mean)

    # Method that update the click probabilities, by counting the user who opened a secondary and dividing it
    # by the number of user who bought the item.
    def update(self, interactions):
        #   - Step 5: updates graph weights.

        visits = np.full(self.num_products, 0)
        bought = np.full(self.num_products, 0)
        openedFirst = np.zeros(self.num_products)
        openedSecond = np.zeros(self.num_products)
        trialFirst = np.zeros(self.num_products)
        trialSecond = np.zeros(self.num_products)

        single_interactions_list = []
        for inter in interactions:
            single_interactions_list.append(inter)
            visits = np.add(visits, inter.linearizeVisits())
            bought = np.add(bought, inter.linearizeBought())
            trialFirst = np.add(trialFirst, inter.linearizeOpenedFirstTrial())
            trialSecond = np.add(trialSecond, inter.linearizeOpenedSecondTrial())

        while len(single_interactions_list) > 0:
            p = single_interactions_list.pop()
            openedFirst[p.product] += p.sec1Opened
            openedSecond[p.product] += p.sec2Opened
            # expand
            for j in range(0, len(p.following)):
                single_interactions_list.append(p.following[j])

        self.trial_click_first = np.add(self.trial_click_first, trialFirst)
        self.trial_click_second = np.add(self.trial_click_second, trialSecond)
        self.times_first_opened = np.add(self.times_first_opened, openedFirst)
        self.times_second_opened = np.add(self.times_second_opened, openedSecond)

        for i in range(0, self.num_products):
            old = self.times_arms_pulled[i][self.configuration[i]]
            self.times_arms_pulled[i][self.configuration[i]] += visits[i]  # Updates the number of times arm is pulled
            mean = self.conversion_rates[i][self.configuration[i]]
            if self.times_arms_pulled[i][self.configuration[i]] > 0:
                self.conversion_rates[i][self.configuration[i]] = (mean * old + bought[i]) / \
                                                              self.times_arms_pulled[i][self.configuration[i]]
            if self.trial_click_first[i] > 0:
                self.clickProbability[i][self.secondary[i][0]] = self.times_first_opened[i] / self.trial_click_first[i]
            if self.trial_click_second[i] > 0:
                self.clickProbability[i][self.secondary[i][1]] = self.times_second_opened[i] / self.trial_click_second[i]
                self.clickProbability[i][self.secondary[i][1]] = self.clickProbability[i][self.secondary[i][1]] / self.Lambda

        if self.debug:
            print("Click Probability: ", self.clickProbability)
            print("Conversion rate: ", self.conversion_rates)
            print(self.trial_click_first, self.trial_click_second, self.times_first_opened, self.times_second_opened)