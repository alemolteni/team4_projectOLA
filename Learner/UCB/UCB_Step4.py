from Learner.UCB.UCB_Step3 import UCB_Step3
from Model.Evaluator.GraphEvaluator import GraphEvaluator
import numpy as np
import math


class UCB_Step4(UCB_Step3):
    def __init__(self, margins=np.ones((5, 4)), num_products=5, num_prices=4, debug=False,
                 clickProbability=np.ones((5, 5)), secondary=None, Lambda=1):
        # Can't use conversion rates and can't use alphas
        self.times_arms_pulled_for_alphas = 0
        self.times_arms_pulled_for_units = np.zeros(num_products)
        self.lower_bound_cr = np.zeros(num_products)
        super(UCB_Step4, self).__init__(margins=margins, num_products=num_products, num_prices=num_prices, debug=debug,
                                        clickProbability=clickProbability, secondary=secondary, Lambda=Lambda)

    # Update alphas counting the number of user that started in each product and dividing it
    # by the total number of users
    # Also updates conversion rates
    def update(self, interactions):
        # From daily interactions extract needed information, depending on step uncertainty:
        #   - Step 4: update conversion rates, 𝛼 ratios, units sold per product

        visits = np.full(self.num_products, 0)
        bought = np.full(self.num_products, 0)
        started = np.full(self.num_products, 0)
        num_units = np.full(self.num_products, 0)

        for inter in interactions:
            visits = np.add(visits, inter.linearizeVisits())
            bought = np.add(bought, inter.linearizeBought())
            started = np.add(started, inter.linearizeStart())
            num_units = np.add(num_units, inter.linearizeNumUnits())

        oldAlpha = self.times_arms_pulled_for_alphas
        self.times_arms_pulled_for_alphas += sum(started)

        for i in range(0, len(self.configuration)):
            oldCR = self.times_arms_pulled[i][self.configuration[i]]
            oldUnits = self.times_arms_pulled_for_units[i]
            self.times_arms_pulled[i][self.configuration[i]] += visits[i]  # Updates number of times arm is pulled
            self.times_arms_pulled_for_units[i] += bought[i]
            meanCR = self.conversion_rates[i][self.configuration[i]]
            meanUnits = self.units_mean[i]
            meanAlpha = self.alphas[i]

            if self.times_arms_pulled[i][self.configuration[i]] > 0:
                self.conversion_rates[i][self.configuration[i]] = (meanCR * oldCR + bought[i]) / \
                                                              self.times_arms_pulled[i][self.configuration[i]]

            if self.times_arms_pulled_for_units[i] > 0:
                self.units_mean[i] = (meanUnits * oldUnits + num_units[i]) / self.times_arms_pulled_for_units[i]

            if self.times_arms_pulled_for_alphas > 0:
                self.alphas[i] = (meanAlpha * oldAlpha + started[i]) / self.times_arms_pulled_for_alphas

        if self.debug:
            print("Conversion rates: ", self.conversion_rates)
            print("Alphas: ", self.alphas, sum(self.alphas))
            print("Num units: ", self.units_mean)
            print("Num units pulled: ", self.times_arms_pulled_for_units)
        return

    # Initialize values using the given interactions
    def batch_update(self, interactions):
        for inter in interactions:
            self.configuration = inter.price_levels
            assert len(self.configuration) is not None
            self.update([inter])
        return

    def compute_product_margin_lower_bound(self):
        log_time_double = np.full((self.num_products, self.num_prices), 2 * math.log(self.t), dtype=float)
        lower_deviation_cr = np.sqrt(np.divide(-np.log(0.05), self.times_arms_pulled,
                                               out=np.full_like(log_time_double, 0, dtype=float),
                                               where=self.times_arms_pulled > 0))
        self.lower_bound_cr = np.subtract(self.conversion_rates, lower_deviation_cr)
        self.lower_bound_cr = np.clip(self.lower_bound_cr, 0, None)

        exp_rewards = np.zeros((self.num_products, self.num_prices))
        for i in range(0, self.num_products):
            for j in range(0, self.num_prices):
                test_config = self.configuration
                test_config[i] = j

                armMargins = []
                armConvRates = []
                # print(self.lower_bound_cr)
                # print(self.times_arms_pulled)
                for k in range(0, len(test_config)):
                    armMargins.append(self.margins[k][test_config[k]])
                    armConvRates.append(self.lower_bound_cr[k][test_config[k]])

                # Units mean doesn't need an upper bound since it doesn't depend on the price of the product
                graphEval = GraphEvaluator(products_list=self.productList, click_prob_matrix=self.clickProbability,
                                           lambda_prob=self.Lambda, alphas=self.alphas,
                                           conversion_rates=armConvRates,
                                           margins=armMargins,
                                           units_mean=self.units_mean, verbose=False, convert_units=False)
                margin = graphEval.computeMargin()

                exp_rewards[i, j] = margin
        # return the best margin testing configuration using a heuristic
        return np.max(exp_rewards)
