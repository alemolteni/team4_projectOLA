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
            if self.verbose: print("Path indexes for product {} : {}".format(self.product, path))
            for k in range(0,len(path)-1):
                prob = prob * self.graph_prob.getEdgeProbability(path[k],path[k+1])
            paths_prob.append(prob)
        # Merge all paths prob by using OR, P(path1 OR path2) = 1 - (1 - P(path1)) * (1 - P(path2))
        inverse_prob = 1
        for i in range(0,len(paths_prob)):
            inverse_prob = inverse_prob * (1 - paths_prob[i])
        return 1 - inverse_prob

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
                next_nodes.append(StepNode(i,paths=self.feasible_paths,graph_prob=self.graph_prob))
        return next_nodes
    
    def isFeasible(self):
        return len(self.feasible_paths) != 0
        