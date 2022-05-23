from Learner.UCB_CR import *


class UCB_Step4(UCB_CR):
    def __init__(self, margins=np.ones((5, 4)), num_products=5, num_prices=4, debug=False,
                 clickProbability=np.ones((5, 5)), secondary=None, Lambda=1):
        # Can't use conversion rates and can't use alphas
        super(UCB_Step4, self).__init__(margins=margins, num_products=num_products, num_prices=num_prices, debug=debug,
                                        clickProbability=clickProbability, secondary=secondary, Lambda=Lambda)

    # Update alphas counting the number of user that started in each product and dividing it
    # by the total number of users
    # Also updates conversion rates
    def update(self, interactions):
        # From daily interactions extract needed information, depending on step uncertainty:
        #   - Step 4: update conversion rates, 𝛼 ratios, #units sold per product

        visits = np.full(self.num_products, 0)
        bought = np.full(self.num_products, 0)
        started = np.full(self.num_products, 0)

        for inter in interactions["episodes"]:
            visits = np.add(visits, inter.linearizeVisits())
            bought = np.add(bought, inter.linearizeBought())
            started = np.add(started, inter.linearizeStart())

        episode_conv_rates = np.divide(bought, visits, out=np.full_like(bought, 0, dtype=float), where=visits != 0)
        episode_alphas = np.divide(started, sum(started))

        for i in range(0, len(self.configuration)):
            # Incremental average m(n+1) = m(n) + (new_val - m(n)) / n+1
            mean_alphas = self.alphas[i]
            mean_cr = self.conversion_rates[i][self.configuration[i]]
            self.conversion_rates[i][self.configuration[i]] = (mean_cr + (episode_conv_rates[i] - mean_cr) / self.t)
            self.alphas[i] = (mean_alphas + (episode_alphas[i] - mean_alphas) / self.t)
        if self.debug:
            print("Conversion rates: ", self.conversion_rates)
            print("Alphas: ", self.alphas, sum(self.alphas))
        return
