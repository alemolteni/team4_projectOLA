from Learner.UCB.UCB_Step4 import UCB_Step4
from Learner.TS_Alphas import TS_Alphas
from ContextTreeNode import ContextTreeNode
import numpy as np


class ContextualLearner:
    def __init__(self, margins=np.ones((5, 4)), num_products=5, num_prices=4, debug=False, approach="ucb",
                 clickProbability=np.ones((5, 5)), secondary=None, Lambda=1, features_names=None, update_frequency=14):

        learner = self.choose_approach(approach, margins=margins, num_products=num_products, num_prices=num_prices,
                                      debug=debug, clickProbability=clickProbability, secondary=secondary,
                                      Lambda=Lambda)

        self.tree = ContextTreeNode(features_names=features_names, learner=learner)
        self.update_frequency = update_frequency
        self.t=0
        self.samples = []

    def choose_approach(self, typeLearner, margins=np.ones((5, 4)), num_products=5, num_prices=4, debug=False,
                 clickProbability=np.ones((5, 5)), secondary=None, Lambda=1):

        typeLearner = typeLearner.lower()
        if typeLearner == "ucb":
            return UCB_Step4(margins=margins, num_products=num_products, num_prices=num_prices, debug=debug,
                                 clickProbability=clickProbability, secondary=secondary, Lambda=Lambda)
        elif typeLearner == "ts":
            return TS_Alphas(margins=margins, num_products=num_products, num_prices=num_prices, debug=debug,
                                 click_prob=clickProbability, secondary_prod=secondary, l=Lambda)

    # Return a set of configuration for each context
    def pull_arm(self):
        leaves = self.tree.get_leaves()
        configs = []
        for leaf in leaves:
            configs.append(leaf.get_learner().pull_arm())
        return configs

    def update(self, interactions):
        self.t += 1
        self.samples.append(interactions)
        if self.t % self.update_frequency == 0:
            self.generate_contexts()
        leaves = self.tree.get_leaves()
        for leaf in leaves:
            leaf.get_learner().update(interactions)
        return

    def generate_contexts(self):
        can_split = []
        for leaf in self.tree.get_leaves():
            if leaf.can_split():
                can_split.append(leaf)
        for leaf in can_split:
            self.evaluate_split(leaf)
        return

    def evaluate_split(self, leaf):
        available_features = list(leaf.features_names - set(leaf.split_features.keys()))
        for feature in available_features:
            # ToDo: Evaluate if the feature needs the split
            print(feature)
        return
