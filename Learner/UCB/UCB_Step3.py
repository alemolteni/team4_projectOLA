from Model.Evaluator.GraphEvaluator import GraphEvaluator
import numpy as np
import math


class UCB_Step3():
    def __init__(self, margins=np.ones((5, 4)), num_products=5, num_prices=4, debug=False, alphas=np.ones(5),
                 clickProbability=np.zeros((5, 5)), secondary=None, Lambda=1, conversion_rates=None, units_mean=None):
        if units_mean is None:
            units_mean = [1, 1, 1, 1, 1]
        if secondary is None:
            secondary = {0: [1, 2], 1: [2, 4], 2: [3, 4], 3: [4, 0],
                         4: [1, 3]}  # Changed secondary syntax to avoid strings

        self.alphas = alphas.copy()
        self.clickProbability = clickProbability.copy()
        self.secondary = []
        self.productList = secondary.copy()
        # Using a different syntax than provided
        for p in secondary:
            self.secondary.append([p.getSecondaryProduct(0), p.getSecondaryProduct(1)])

        self.Lambda = Lambda
        self.configuration = [0 for i in range(0, num_products)]
        self.t = 0
        self.debug = debug
        self.num_products = num_products
        self.num_prices = num_prices

        if conversion_rates is None:
            self.conversion_rates = np.zeros((self.num_products, self.num_prices))
        else:
            self.conversion_rates = conversion_rates.copy()
        self.times_arms_pulled = np.full((num_products, num_prices), 0.0)
        self.expected_reward = np.full((self.num_products, self.num_prices), 0.0)
        self.margins = margins.copy()
        self.units_mean = units_mean.copy()

        self.S = 1
        self.upper_bound_cr = np.zeros((self.num_products, self.num_prices))

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
            log_time = np.full((self.num_products, self.num_prices), 2 * math.log(self.t), dtype=float)
            upper_deviation = np.sqrt(np.divide(log_time, self.times_arms_pulled,
                                                out=np.full_like(log_time, 0, dtype=float),
                                                where=self.times_arms_pulled > 0))

            self.upper_bound_cr = np.add(self.conversion_rates, upper_deviation)

            self.expected_reward = self.compute_expected_reward()

            self.configuration = np.argmax(self.expected_reward, axis=1)

        if self.debug:
            print("Config: ", self.configuration)
            print("Times arms pulled: ", self.times_arms_pulled)
            if self.t > 4:
                print("Upper_deviation: ", upper_deviation)
            print("Expected rew: ", self.expected_reward)
            sum = 0
            for i in range(0, 4):
                sum += self.expected_reward[i][self.configuration[i]] * self.alphas[i]
            print("Sum: ", sum)
        return self.configuration

    def update(self, interactions):
        # From daily interactions extract needed information, depending on step uncertainty:
        #   - Step 3: update belief over conversion rates

        visits = np.full(self.num_products, 0)
        bought = np.full(self.num_products, 0)

        for inter in interactions:
            visits = np.add(visits, inter.linearizeVisits())
            bought = np.add(bought, inter.linearizeBought())

        for i in range(0, len(self.configuration)):
            old = self.times_arms_pulled[i][self.configuration[i]]
            self.times_arms_pulled[i][self.configuration[i]] += visits[i]  # Updates the number of times arm is pulled
            mean = self.conversion_rates[i][self.configuration[i]]
            if self.times_arms_pulled[i][self.configuration[i]] > 0:
                self.conversion_rates[i][self.configuration[i]] = (mean * old + bought[i]) / \
                                                              self.times_arms_pulled[i][self.configuration[i]]

        if self.debug:
            print("Conversion rates: ", self.conversion_rates)
        return

    def compute_expected_reward(self):
        # It should return a matrix #PROD x #LEVELS in which the elements are the computed rewards
        # Then pull arm will use this to choose the arm with the max expected reward as next

        exp_rewards = np.zeros((self.num_products, self.num_prices))
        for i in range(0, self.num_products):
            for j in range(0, self.num_prices):
                test_config = self.configuration
                test_config[i] = j
                margin = self.compute_product_margin(test_config)
                exp_rewards[i, j] = margin
        return exp_rewards

    def compute_product_margin(self, test_config):
        armMargins = []
        armConvRates = []
        for k in range(0, len(test_config)):
            armMargins.append(self.margins[k][test_config[k]])
            armConvRates.append(self.upper_bound_cr[k][test_config[k]])

        graphEval = GraphEvaluator(products_list=self.productList, click_prob_matrix=self.clickProbability,
                                   lambda_prob=self.Lambda, alphas=self.alphas, conversion_rates=armConvRates,
                                   margins=armMargins,
                                   units_mean=self.units_mean, verbose=False, convert_units=False)
        margin = graphEval.computeMargin()
        return margin