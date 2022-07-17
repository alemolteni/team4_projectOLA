import numpy as np


class GreedyLearner:

    def __init__(self, debug=False):
        self.configuration_margin = [0, 0, 0, 0, 0]
        self.t = -1  # -1 means that the learner is not initialized
        self.optimal_configuration = [0, 0, 0, 0, 0]
        self.optimal_configuration_margin = 0
        self.isOptima = False
        self.numberOfComparison = 5
        self.debug = debug

    def pull_arm(self):
        if self.t == -1 or self.isOptima:
            return self.optimal_configuration  # initial configuration, then updated to be optimal

        try_configuration = self.optimal_configuration.copy()

        # Can't try arm above 3
        if try_configuration[self.t] < 3:
            try_configuration[self.t] += 1

        return try_configuration

    def update(self, margin):
        # Initialization
        if self.t == -1:
            self.optimal_configuration_margin = margin
            self.t += 1
            return

        # Set margin to 0 if the arm was not changed, so if the optimal configuration arm is already at 3
        # This is done to not compare again the optimal configuration with the other derivation, since if the arm is at
        # 3 it was not increased in the pull arm phase.
        if self.optimal_configuration[self.t] != 3:
            self.configuration_margin[self.t] = margin
        else:
            self.configuration_margin[self.t] = 0

        self.t += 1

        # If t==5 we have tried all derivation and must update the optimal margin and configuration (if any are better)
        if self.t == 5:
            result = np.where(self.configuration_margin == np.amax(self.configuration_margin))
            result = result[0]
            result = np.random.choice(result)

            if np.amax(self.configuration_margin) > self.optimal_configuration_margin and not self.isOptima:
                if self.optimal_configuration[result] < 3:
                    self.optimal_configuration[result] += 1
                self.optimal_configuration_margin = np.amax(self.configuration_margin)
                if self.debug:
                    print(self.configuration_margin, self.optimal_configuration_margin, self.optimal_configuration)
            else:
                self.isOptima = True
                if self.debug:
                    print(self.configuration_margin, self.optimal_configuration_margin, self.optimal_configuration)
            self.t = 0
