import numpy as np

from Learner.UCB_CR import *


class UCB_Step5(UCB_CR):
    def __init__(self, margins=np.ones((5, 4)), num_products=5, num_prices=4, debug=False,
                 alphas=np.zeros(5), conversion_rates=np.zeros((5, 4)), secondary=None, Lambda=1):
        # Can't use click probability
        super(UCB_Step5, self).__init__(margins=margins, num_products=num_products, num_prices=num_prices, debug=debug,
                                        alphas=alphas, secondary=secondary, Lambda=Lambda,
                                        conversion_rates=conversion_rates)

    # Method that update the click probabilities, by counting the user who opened a secondary and dividing it
    # by the number of user who bought the item.
    def update(self, interactions):
        #   - Step 5: updates graph weights, now they are underestimated, don't know if it is correct.

        openedFirst = np.zeros(self.num_products)
        openedSecond = np.zeros(self.num_products)
        tot = np.zeros(self.num_products)

        single_interactions_list = []
        for i in interactions["episodes"]:
            single_interactions_list.append(i)

        while len(single_interactions_list) > 0:
            p = single_interactions_list.pop(0)

            # update
            if p.bought:
                tot[p.product] += 1

                # first element
                if p.sec1Opened:
                    openedFirst[p.product] += 1

                # second element
                if p.sec2Opened:
                    openedSecond[p.product] += 1

            # expand
            for j in range(0, len(p.following)):
                single_interactions_list.append(p.following[j])

        episode_clickProb = np.zeros((self.num_products, self.num_products))

        t = 0
        for key in self.secondary:
            if tot[t] != 0:
                episode_clickProb[t][key[0]] = openedFirst[t] / tot[t]
                episode_clickProb[t][key[1]] = openedSecond[t] / (tot[t] * self.Lambda)
            t += 1

        for i in range(0, self.num_products):  # Weighting the mean by the number of sample could have sense?
            for j in range(0, self.num_products):
                mean = self.clickProbability[i][j]
                self.clickProbability[i][j] = (mean + (episode_clickProb[i][j] - mean) / self.t)

        if self.debug:
            print("Click Probability: ", self.clickProbability)
