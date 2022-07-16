import numpy as np
from Learner.TS import *
from Learner.TS_CR import *


class TS_Alphas(TS_CR):

    def __init__(self, num_products=5, num_prices=4, margins=np.ones((5, 4)), secondary_prod=[], l=0.5,
                 click_prob=np.ones((5, 5)), debug=False):
        super(TS_Alphas, self).__init__(num_products=num_products, num_prices=num_prices, margins=margins,
                                        click_prob=click_prob, l=l, secondary_prod=secondary_prod)
        self.starting_cumulative = np.full(self.num_products, 0)
        self.units_cumulative = np.full(self.num_products, 0, dtype=float)
        self.total_times_bought = np.full(self.num_products, 0, dtype=float)

        self.alphas = np.full(self.num_products, 1/self.num_products)
        self.units_mean = np.full(self.num_products, 1)
        self.secondary_prod = secondary_prod

    # alphas parameters are treated as the mean of a beta distribution
    

    def pull_arm(self):
        arm = super(TS_Alphas, self).pull_arm()

        log_time_double = np.full((self.num_products, self.num_prices), 2 * math.log(self.t), dtype=float)
        lower_deviation_cr = np.sqrt(np.divide(-np.log(0.05), self.times_arms_pulled,
                                               out=np.full_like(log_time_double, 0, dtype=float),
                                               where=self.times_arms_pulled > 0))
        def from_beta_to_cr(a):
            return a[0] / (a[0] + a[1])
        expected_cr = np.apply_along_axis(from_beta_to_cr, 2, self.conversion_rates_distro)
                
        self.lower_bound_cr = np.subtract(expected_cr, lower_deviation_cr)
        self.lower_bound_cr = np.clip(self.lower_bound_cr, 0, None)

        return arm

    def update(self, interactions):
        # From daily interactions extract needed information, depending on step uncertainty:
        #   - Step 4: update conversion rates, 𝛼 ratios, #units sold per product

        super(TS_Alphas, self).update(interactions)

        for inter in interactions:
            self.starting_cumulative = np.add(self.starting_cumulative, inter.linearizeStart())
            self.units_cumulative = np.add(self.units_cumulative, inter.linearizeNumUnits())
            self.total_times_bought = np.add(self.total_times_bought, inter.linearizeBought())
        
        self.alphas = self.starting_cumulative / self.starting_cumulative.sum()
        
        # https://stackoverflow.com/questions/26248654/how-to-return-0-with-divide-by-zero
        # c = np.divide(a, b, out=np.full_like(a, 1), where=b!=0)
        self.units_mean = np.divide(self.units_cumulative, self.total_times_bought, 
                                    out=np.full_like(self.units_cumulative, 1), where=self.total_times_bought!=0)
        return

    def batch_update(self, interactions):
        for inter in interactions:
            self.configuration = inter.price_levels
            assert len(self.configuration) is not None
            self.update([inter])

    def compute_product_margin_lower_bound(self):
        self.pull_arm()  # Update the optimal configuration
        self.t -= 1  # Don't count the pull of the arm as a step
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
                graphEval = GraphEvaluator(products_list=self.secondary_prod, click_prob_matrix=self.click_prob,
                                           lambda_prob=self.l, alphas=self.alphas,
                                           conversion_rates=armConvRates,
                                           margins=armMargins,
                                           units_mean=self.units_mean, verbose=False, convert_units=False)
                margin = graphEval.computeMargin()

                exp_rewards[i, j] = margin

        test_config = np.argmax(exp_rewards, axis=1)
        armMargins = []
        armConvRates = []
        # print(self.lower_bound_cr)
        # print(self.times_arms_pulled)
        for k in range(0, len(test_config)):
            armMargins.append(self.margins[k][test_config[k]])
            armConvRates.append(self.lower_bound_cr[k][test_config[k]])

        graphEval = GraphEvaluator(products_list=self.productList, click_prob_matrix=self.clickProbability,
                                   lambda_prob=self.Lambda, alphas=self.alphas,
                                   conversion_rates=armConvRates,
                                   margins=armMargins,
                                   units_mean=self.units_mean, verbose=False, convert_units=False)
        margin = graphEval.computeMargin()
        # return the best margin testing configuration using a heuristic
        return margin
