import copy

# Represent a node in the context tree
class ContextTreeNode:
    def __init__(self, features_names, learner):
        # Each node has its own learner
        self.left = None  # Consider the node feature as 0
        self.right = None  # Consider the node feature as 1
        self.features_names = features_names
        self.learner = learner
        self.split_features: dict = {}

    def __str__(self):
        if self.is_leaf():
            return "Leaf: " + str(self.split_features)
        else:
            return "Node: " + str(self.split_features) + " Left " + str(self.left) + " Right " + str(self.right)

    def is_leaf(self):
        return self.left is None and self.right is None

    def get_leaves(self):
        if self.is_leaf():
            return [self]
        else:
            return self.left.get_leaves() + self.right.get_leaves()

    def split(self, split_feature, left_learner, right_learner):
        self.left = ContextTreeNode(self.features_names, left_learner)
        self.left.split_features = copy.deepcopy(self.split_features)
        self.left.split_features[split_feature] = 0
        self.right = ContextTreeNode(self.features_names, right_learner)
        self.right.split_features = copy.deepcopy(self.split_features)
        self.right.split_features[split_feature] = 1

    def get_learner(self):
        return self.learner

    def pull_arm(self):
        return self.learner.pull_arm()

    def update(self, interactions):
        return self.learner.update(interactions)

    # len(self.features_names) is the maximum height of the tree (starting from 0)
    def can_split(self):
        return len(self.split_features) < len(self.features_names)


paolo = ContextTreeNode(features_names=['a', 'b'], learner=None)
paolo.split('a', None, None)
paola = paolo.left
paola.split('b', None, None)
print(paolo)