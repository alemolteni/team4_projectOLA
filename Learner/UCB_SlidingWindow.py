from Learner.UCB_CR import *
import numpy as np


class UCB_SlidingWindow(UCB_CR):

    def __init__(self, margins=np.ones((5, 4)), num_products=5, num_prices=4, debug=False, alphas=np.zeros(5),
                 clickProbability=np.zeros((5, 5)), secondary=None, Lambda=1, sliding_window_size=20):
        # Can't use conversion rates and can't use alphas
        self.sw_size = sliding_window_size
        self.circular_buffer_cr = [[]] * sliding_window_size
        self.circular_buffer_config = [[]] * sliding_window_size
        self.headPoint = 0

        self.initializeBuffer = True
        super(UCB_SlidingWindow, self).__init__(margins=margins, num_products=num_products, num_prices=num_prices,
                                                debug=debug,
                                                clickProbability=clickProbability, secondary=secondary, Lambda=Lambda,
                                                alphas=alphas)

    def pull_arm(self):
        # Choose arm with higher upper confidence bound
        # UCB sliding window is defined as arm ← argmax{a∈A} (x(a.tau) + sqrt(2*log(t)/n(a,t − 1, tau))
        # x(a.tau) is the expected reward on the sliding window
        self.t += 1
        log_time = np.full((self.num_products, self.num_prices), 2 * math.log(self.t), dtype=float)
        n_arms = self.times_arms_pulled
        upper_deviation = np.sqrt(
            np.divide(log_time, n_arms, out=np.full_like(log_time, np.inf, dtype=float), where=n_arms != 0))

        self.expected_reward = self.compute_expected_rewards()
        if self.debug:
            print("Reward: ", self.expected_reward)
        upper_bound = np.add(self.expected_reward, upper_deviation)

        self.configuration = np.argmax(upper_bound, axis=1)
        old = self.circular_buffer_config[self.headPoint]
        self.circular_buffer_config[self.headPoint] = self.configuration.copy()
        for i in range(0, len(self.configuration)):
            self.times_arms_pulled[i][self.configuration[i]] += 1
            if not self.initializeBuffer:
                self.times_arms_pulled[i][old[i]] -= 1  # Forget oldest configuration used
        return self.configuration

    def update(self, interactions):
        # From daily interactions extract needed information, depending on step uncertainty:
        #   - Step 6: update belief over conversion rates using a sliding window

        visits = np.full(self.num_products, 0)
        bought = np.full(self.num_products, 0)

        for inter in interactions:
            visits = np.add(visits, inter.linearizeVisits())
            bought = np.add(bought, inter.linearizeBought())

        episode_conv_rates = np.divide(bought, visits, out=np.full_like(bought, 0, dtype=float), where=visits != 0)
        episode_conv_rates_matrix = np.zeros((self.num_products, self.num_prices))

        for i in range(0, len(self.configuration)):
            episode_conv_rates_matrix[i][self.configuration[i]] = episode_conv_rates[i]

        if self.initializeBuffer:
            self.circular_buffer_cr[self.headPoint] = episode_conv_rates_matrix.copy()
            for i in range(0, len(self.configuration)):
                # Incremental average m(n+1) = m(n) + (new_val - m(n)) / n+1
                mean = self.conversion_rates[i][self.configuration[i]]
                self.conversion_rates[i][self.configuration[i]] = (mean + (episode_conv_rates[i] - mean) / self.t)

            self.headPoint += 1
            if self.headPoint >= self.sw_size:
                self.initializeBuffer = False
                self.headPoint = 0
            #  print(self.conversion_rates)

        else:
            #  oldData = self.circular_buffer[self.headPoint].copy()
            self.circular_buffer_cr[self.headPoint] = episode_conv_rates_matrix.copy()

            temp_conversion_rates = np.zeros((self.num_products, self.num_prices))

            for i in range(0, self.sw_size):
                temp_conversion_rates = np.add(temp_conversion_rates, self.circular_buffer_cr[i])

            temp_conversion_rates = np.divide(temp_conversion_rates, self.sw_size)

            if self.headPoint < self.sw_size - 1:
                self.headPoint += 1
            else:
                self.headPoint = 0

            self.conversion_rates = temp_conversion_rates.copy()

        print("Conversion rates: ", self.conversion_rates)
