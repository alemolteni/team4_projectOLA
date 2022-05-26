import numpy as np

class StepNode:
    def __init__(self, product, paths=[], graph_prob=None, verbose=False):
        self.paths=paths
        self.product = product
        self.feasible_paths = []
        self.graph_prob = graph_prob
        self.verbose = verbose
        productIndex = product
        for i in range(0,len(paths)):
            path = np.array(paths[i])
            if np.any(path[:] == product) == False:
                path = np.append(path, productIndex)
                self.feasible_paths.append(path.tolist())
        
        
    def computeProbability(self):
        paths_prob = []
        # Compute probability of each path
        for i in range(0,len(self.feasible_paths)):
            path = self.feasible_paths[i]
            prob = 1
            for k in range(0,len(path)-1):
                prob = prob * self.graph_prob.getEdgeProbability(path[k],path[k+1])

            if self.verbose: print("Path indexes for product {}: {} with probability {}".format(self.product, path, prob))
            paths_prob.append(prob)
        # Merge all paths prob by using OR, P(path1 OR path2) = 1 - (1 - P(path1)) * (1 - P(path2))
        # UPDATED: Actually no bc P(path1 OR path2) = P(path1) + P(path2) - P(path1 AND path2) = P(path1) + P(path2)
        # by problem construction, in each interaction we can reach a product in a single way (can open a product once)
        return np.array(paths_prob).sum()

    def merge(self, node):
        if self.product != node.product:
            return
        # Concatenate the two lists
        self.feasible_paths += node.feasible_paths
        return self

    def expand(self):
        n_products = self.graph_prob.shape()[0]
        next_nodes = []
        for i in range(0,n_products):
            if self.graph_prob.getEdgeProbability(self.product,i) > 0 :
                next_nodes.append(StepNode(i,paths=self.feasible_paths,graph_prob=self.graph_prob,verbose=self.verbose))
        return next_nodes
    
    def isFeasible(self):
        return len(self.feasible_paths) != 0
        