
from Learner.UCB.UCB_Step3 import *


class UCB_Step5(UCB_Step3):
    def __init__(self, margins=np.ones((5, 4)), num_products=5, num_prices=4, debug=False,
                 alphas=np.zeros(5), conversion_rates=np.zeros((5, 4)), secondary=None, Lambda=1, units_mean=None):
        # Can't use click probability
        self.times_arms_pulled_click = np.zeros((num_products))
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
            log_time_click = np.full((self.num_products, self.num_products), 2 * math.log(self.t), dtype=float)

            log_time_cr = np.full((self.num_products, self.num_prices), 2 * math.log(self.t), dtype=float)

            upper_deviation_click = np.sqrt(np.divide(log_time_click, self.times_arms_pulled_click,
                                                      out=np.full_like(log_time_click, 0, dtype=float),
                                                      where=self.times_arms_pulled_click != 0))

            upper_deviation_cr = np.sqrt(np.divide(log_time_cr, self.times_arms_pulled,
                                                out=np.full_like(log_time_cr, 0, dtype=float),
                                                where=self.times_arms_pulled > 0))

            self.upper_bound_click = np.add(self.clickProbability, upper_deviation_click)

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

        single_interactions_list = []
        for inter in interactions:
            single_interactions_list.append(inter)
            visits = np.add(visits, inter.linearizeVisits())
            bought = np.add(bought, inter.linearizeBought())

        while len(single_interactions_list) > 0:
            p = single_interactions_list.pop(0)

            # update
            if p.bought == 1:

                # first element
                if p.sec1Opened == 1:
                    openedFirst[p.product] += 1

                # second element
                if p.sec2Opened == 1:
                    openedSecond[p.product] += 1

            # expand
            for j in range(0, len(p.following)):
                single_interactions_list.append(p.following[j])
        self.times_arms_pulled_click = np.add(self.times_arms_pulled_click, bought)
        self.times_first_opened = np.add(self.times_first_opened, openedFirst)
        self.times_second_opened = np.add(self.times_second_opened, openedSecond)

        for i in range(0, self.num_products):
            old = self.times_arms_pulled[i][self.configuration[i]]
            self.times_arms_pulled[i][self.configuration[i]] += visits[i]  # Updates the number of times arm is pulled
            mean = self.conversion_rates[i][self.configuration[i]]
            self.conversion_rates[i][self.configuration[i]] = (mean * old + bought[i]) / \
                                                              self.times_arms_pulled[i][self.configuration[i]]

            self.clickProbability[i][self.secondary[i][0]] = self.times_first_opened[i] / self.times_arms_pulled_click[i]
            self.clickProbability[i][self.secondary[i][1]] = self.times_second_opened[i] / self.times_arms_pulled_click[i]
            self.clickProbability[i][self.secondary[i][1]] = self.clickProbability[i][self.secondary[i][1]] / self.Lambda

        if self.debug:
            print("Click Probability: ", self.clickProbability)
            print("Times arms pulled click: ", self.times_arms_pulled_click)

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