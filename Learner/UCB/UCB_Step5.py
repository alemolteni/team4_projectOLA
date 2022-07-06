import numpy as np

from Learner.UCB.UCB_Step3 import *


class UCB_Step5(UCB_Step3):
    def __init__(self, margins=np.ones((5, 4)), num_products=5, num_prices=4, debug=False,
                 alphas=np.zeros(5), conversion_rates=np.zeros((5, 4)), secondary=None, Lambda=1, units_mean=None):
        # Can't use click probability
        self.trial_click_first = np.zeros((num_products))
        self.trial_click_second = np.zeros((num_products))
        self.times_first_opened = np.zeros((num_products))
        self.times_second_opened = np.zeros((num_products))
        self.upper_bound_click = np.zeros((num_products, num_products))
        super(UCB_Step5, self).__init__(margins=margins, num_products=num_products, num_prices=num_prices, debug=debug,
                                        alphas=alphas, secondary=secondary, Lambda=Lambda,
                                        conversion_rates=conversion_rates, units_mean=units_mean)

    def pull_arm(self):
        # Choose arm with higher upper confidence bound
        # UCB is defined as arm ← argmax{a∈A} (x(a) + sqrt(2*log(t)/n(a,t − 1))
        # x(a) is the expected reward till now (which is the metric for the reward?)
        self.t += 1
        # Run every arm at least once
        if self.t <= 4:
            self.configuration = [self.t - 1, self.t - 1, self.t - 1, self.t - 1, self.t - 1]
        # Choose the arm with the highest upper bound
        else:
            log_time_click = np.full(self.num_products, 2 * math.log(self.t), dtype=float)

            log_time_cr = np.full((self.num_products, self.num_prices), 2 * math.log(self.t), dtype=float)

            upper_deviation_click_first = np.sqrt(np.divide(log_time_click, self.trial_click_first,
                                                      out=np.full_like(log_time_click, 0, dtype=float),
                                                      where=self.trial_click_first != 0))

            upper_deviation_click_second = np.sqrt(np.divide(log_time_click, self.trial_click_second,
                                                      out=np.full_like(log_time_click, 0, dtype=float),
                                                      where=self.trial_click_second != 0))

            upper_deviation_cr = np.sqrt(np.divide(log_time_cr, self.times_arms_pulled,
                                                out=np.full_like(log_time_cr, 0, dtype=float),
                                                where=self.times_arms_pulled > 0))

            for prod in range(0, self.num_products):
                self.upper_bound_click[prod][self.secondary[prod][0]] = self.clickProbability[prod][self.secondary[prod][0]] +\
                                                                        upper_deviation_click_first[prod]
                self.upper_bound_click[prod][self.secondary[prod][1]] = self.clickProbability[prod][self.secondary[prod][1]] +\
                                                                        upper_deviation_click_second[prod]

            self.upper_bound_cr = np.add(self.conversion_rates, upper_deviation_cr)

            self.expected_reward = self.compute_expected_reward()

            self.configuration = np.argmax(self.expected_reward, axis=1)

        if self.debug:
            print("Config: ", self.configuration)
            print("Times arms pulled: ", self.times_arms_pulled)
            #if self.t > 4: print("Upper_deviation: ", upper_deviation)
            print("Expected rew: ", self.expected_reward)
            print("CR: ", self.conversion_rates)
        return self.configuration

    # Method that update the click probabilities, by counting the user who opened a secondary and dividing it
    # by the number of user who bought the item.
    def update(self, interactions):
        #   - Step 5: updates graph weights, now they are underestimated, don't know if it is correct.

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

    def compute_product_margin(self, test_config):
        armMargins = []
        armConvRates = []
        for k in range(0, len(test_config)):
            armMargins.append(self.margins[k][test_config[k]])
            armConvRates.append(self.upper_bound_cr[k][test_config[k]])

        graphEval = GraphEvaluator(products_list=self.productList, click_prob_matrix=self.upper_bound_click,
                                   lambda_prob=self.Lambda, alphas=self.alphas,
                                   conversion_rates=armConvRates,
                                   margins=armMargins,
                                   units_mean=self.units_mean, verbose=False, convert_units=False)
        margin = graphEval.computeMargin()
        return margin