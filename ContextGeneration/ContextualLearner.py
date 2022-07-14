import copy

from Learner.UCB.UCB_Step4 import UCB_Step4
from Learner.TS_Alphas import TS_Alphas
from ContextGeneration.ContextTreeNode import ContextTreeNode
import numpy as np


class ContextualLearner:
    def __init__(self, margins=np.ones((5, 4)), num_products=5, num_prices=4, debug=False, approach="ts",
                 clickProbability=None, secondary=None, Lambda=1, features_names=None, update_frequency=14):

        self.margins = margins
        self.num_products = num_products
        self.num_prices = num_prices
        self.debug = debug
        self.approach = approach
        self.secondary = secondary
        self.Lambda = Lambda

        # This is an array of [click_prob, features, n_user] for each user class
        self.clickProbability = clickProbability

        cp = self.weight_click_probs({})
        learner = self.choose_approach(self.approach, margins=self.margins, num_products=self.num_products,
                                       num_prices=self.num_prices, debug=self.debug,
                                       clickProbability=cp, secondary=self.secondary,
                                       Lambda=self.Lambda)

        self.features_names = features_names
        self.tree = ContextTreeNode(features_names=features_names, learner=learner)
        self.update_frequency = update_frequency
        self.t = 0
        self.samples = []

    # Return a weighted average of the click probability for the specified context: {feature: value}
    def weight_click_probs(self, context):
        # print(context)
        cp = np.zeros((self.num_products, self.num_products), dtype=np.float)
        n = 0
        if len(context) == 0:
            for i in range(0, len(self.clickProbability)):
                cp = np.add(cp, np.array(self.clickProbability[i][0]) * self.clickProbability[i][2])
                n += self.clickProbability[i][2]
            return np.divide(cp, n)
        elif len(context) == 1:
            for i in range(0, len(self.clickProbability)):
                for f in context.keys():
                    if context[f] == self.clickProbability[i][1][f]:
                        cp = np.add(cp, np.array(self.clickProbability[i][0]) * self.clickProbability[i][2])
                        n += self.clickProbability[i][2]
            return np.divide(cp, n)
        elif len(context) == 2:
            for i in range(0, len(self.clickProbability)):
                if context == self.clickProbability[i][1]:
                    return self.clickProbability[i][0]

        raise Exception("Context len impossible")


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
            configs.append([leaf.get_learner().pull_arm(), leaf.split_features])
        return configs

    def update(self, interactions):
        self.t += 1
        self.samples = np.concatenate((self.samples, interactions))
        if self.t % self.update_frequency == 0:
            self.generate_contexts()
        leaves = self.tree.get_leaves()
        for leaf in leaves:
            good_interactions = self.select_compliant_samples(interactions, leaf)
            leaf.get_learner().update(good_interactions)
        return

    def select_compliant_samples(self, interactions, leaf):
        # Select only interaction compliant with the feature of the node
        samples = []
        for sample in interactions:
            compliant = True
            for f in leaf.split_features:
                idx = -1
                for j in range(0, len(self.features_names)):
                    if self.features_names[j] == f:
                        idx = j
                if idx == -1:
                    raise Exception("Feature of the leaf not found")
                if sample.featuresValues[idx] != leaf.split_features[f]:
                    compliant = False
            if compliant:
                samples.append(sample)
        return samples

    def generate_contexts(self):
        if self.debug: print("Before tree: ", self.tree)
        can_split = []
        for leaf in self.tree.get_leaves():
            if leaf.can_split():
                can_split.append(leaf)
        for leaf in can_split:
            if self.evaluate_split(leaf):
                if self.debug: print("After tree: ", self.tree)
                break
        return

    def evaluate_split(self, leaf):
        #print(leaf.split_features)
        available_features = list(set(leaf.features_names) - set(leaf.split_features.keys()))
        for feature in available_features:
            if self.debug: print(feature)
            feature_id = self.features_names.index(feature)

            # Select only interaction compliant with the feature of the node
            samples = self.select_compliant_samples(self.samples, leaf)
            left_split_samples = []
            right_split_samples = []

            for sample in samples:
                if sample.featuresValues[feature_id] == 0:
                    left_split_samples.append(sample)
                else:
                    right_split_samples.append(sample)

            # Evaluate the split
            left_split_probabilities_lb = 0
            right_split_probabilities_lb = 0
            if len(samples) != 0 and len(left_split_samples) != 0:
                left_split_probabilities = len(left_split_samples) / len(samples)
                left_split_probabilities_lb = left_split_probabilities - np.sqrt(-np.log(0.95) / (2*len(left_split_samples)))
            if len(samples) != 0 and len(right_split_samples) != 0:
                right_split_probabilities = len(right_split_samples) / len(samples)
                right_split_probabilities_lb = right_split_probabilities - np.sqrt(-np.log(0.95) / (2*len(right_split_samples)))

            # print(len(samples), len(left_split_samples), len(right_split_samples))
            if len(samples) != 0 and len(left_split_samples) != 0 and len(right_split_samples) != 0:
                assert left_split_probabilities + right_split_probabilities == 1


            # print("How many samples: ", len(left_split_samples), len(right_split_samples), len(samples))
            new_leaf_split_r = copy.deepcopy(leaf.split_features)
            new_leaf_split_l = copy.deepcopy(leaf.split_features)
            new_leaf_split_l[feature] = 0
            new_leaf_split_r[feature] = 1

            leftLearner = self.choose_approach(self.approach, margins=self.margins, num_products=self.num_products,
                                               num_prices=self.num_prices, debug=self.debug,
                                               clickProbability=self.weight_click_probs(new_leaf_split_l), secondary=self.secondary,
                                               Lambda=self.Lambda)
            rightLearner = self.choose_approach(self.approach, margins=self.margins, num_products=self.num_products,
                                                num_prices=self.num_prices, debug=self.debug,
                                                clickProbability=self.weight_click_probs(new_leaf_split_r), secondary=self.secondary,
                                                Lambda=self.Lambda)

            leftLearner.batch_update(left_split_samples)
            leftLearner.t = self.t
            rightLearner.batch_update(right_split_samples)
            rightLearner.t = self.t

            left_margin = leftLearner.compute_product_margin_lower_bound()
            right_margin = rightLearner.compute_product_margin_lower_bound()
            # print("Left margin: ", left_margin, "\n", leftLearner.lower_bound_cr)
            # print("Right margin: ", right_margin, "\n" , rightLearner.lower_bound_cr)

            previous_margin = leaf.get_learner().compute_product_margin_lower_bound()

            if self.debug: print(left_margin * left_split_probabilities_lb, right_margin * right_split_probabilities_lb,
                  previous_margin)

            if left_margin * left_split_probabilities_lb + right_margin * right_split_probabilities_lb > previous_margin:
                leaf.split(feature, leftLearner, rightLearner)
                # print(leaf.split_features)
                return True
        return False
