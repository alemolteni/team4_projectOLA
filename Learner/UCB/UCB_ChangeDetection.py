from Learner.UCB.UCB_Step3 import *
import random


class UCB_ChangeDetection(UCB_Step3):
    def __init__(self, margins=np.ones((5, 4)), num_products=5, num_prices=4, debug=False, alphas=np.zeros(5),
                 clickProbability=np.zeros((5, 5)), secondary=None, Lambda=1, conversion_rates=None, m=10, eps=0.05,
                 h=1, alpha=0.05):
        # Can't use conversion rates and can't use alphas
        super(UCB_ChangeDetection, self).__init__(margins=margins, num_products=num_products, num_prices=num_prices,
                                                  debug=debug,
                                                  clickProbability=clickProbability, secondary=secondary, Lambda=Lambda,
                                                  alphas=alphas, conversion_rates=conversion_rates)
        self.upper_deviation = np.zeros((self.num_products, self.num_prices))
        self.m = m
        self.eps = eps
        self.h = h
        self.g_plus = np.zeros((self.num_products, self.num_prices))
        self.g_minus = np.zeros((self.num_products, self.num_prices))
        self.alpha = alpha

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
            if np.random.binomial(1, 1 - self.alpha):
                print("\nregular")
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
            else:
                print("\nrandom")
                self.configuration[0] = random.randint(0, 3)
                self.configuration[1] = random.randint(0, 3)
                self.configuration[2] = random.randint(0, 3)
                self.configuration[3] = random.randint(0, 3)
                self.configuration[4] = random.randint(0, 3)

        return self.configuration

    def update(self, interactions):
        # From daily interactions extract needed information, depending on step uncertainty:
        #   - Step 3: update belief over conversion rates

        visits = np.full(self.num_products, 0)
        bought = np.full(self.num_products, 0)

        for inter in interactions:
            visits = np.add(visits, inter.linearizeVisits())
            bought = np.add(bought, inter.linearizeBought())

        if self.update_cumulative_sum(bought / visits):
            print("\n\nCHANGE\n\n")
            self.reset()
        else:
            for i in range(0, len(self.configuration)):
                old = self.times_arms_pulled[i][self.configuration[i]]
                self.times_arms_pulled[i][self.configuration[i]] += visits[i]  # Updates the number of times arm is pulled
                mean = self.conversion_rates[i][self.configuration[i]]
                self.conversion_rates[i][self.configuration[i]] = (mean * old + bought[i]) / \
                                                                  self.times_arms_pulled[i][self.configuration[i]]

        if self.debug:
            print("Conversion rates: ", self.conversion_rates)
        return

    def update_cumulative_sum(self, sample):
        if self.t < self.m:
            return 0
        else:
            detection = 0
            for i in range(0, self.num_products):
                s_plus = (sample[i] - self.conversion_rates[i][self.configuration[i]]) - self.eps
                s_minus = - (sample[i] - self.conversion_rates[i][self.configuration[i]]) - self.eps
                self.g_plus[i][self.configuration[i]] = max(0, self.g_plus[i][self.configuration[i]] + s_plus)
                self.g_minus[i][self.configuration[i]] = max(0, self.g_minus[i][self.configuration[i]] + s_minus)
                if self.g_plus[i][self.configuration[i]] > self.h or self.g_minus[i][self.configuration[i]] > self.h:
                    detection = 1
                    i = self.num_products

            return detection

    def reset(self):
        self.g_plus = np.zeros((self.num_products, self.num_prices))
        self.g_minus = np.zeros((self.num_products, self.num_prices))
        self.conversion_rates = np.zeros((self.num_products, self.num_prices))
        self.upper_deviation = np.zeros((self.num_products, self.num_prices))
        self.times_arms_pulled = np.zeros((self.num_products, self.num_prices))
        self.expected_reward = np.zeros((self.num_products, self.num_prices))
        self.upper_bound_cr = np.zeros((self.num_products, self.num_prices))
        self.t = 0

