from Model.GraphProbabilities import GraphProbabilities
from Model.Evaluator.StepNode import StepNode
import numpy as np


class GraphEvaluator:
    def __init__(self, products_list=[], click_prob_matrix=None, lambda_prob=0.5, conversion_rates=[], 
                alphas=[], margins=[], verbose=False):
        assert len(products_list) == len(conversion_rates) and len(products_list) == len(alphas)
        assert len(products_list) == len(margins)
        assert click_prob_matrix is not None

        self.products_list = products_list
        self.lambda_prob = lambda_prob
        self.conversion_rates = np.array(conversion_rates)
        self.n_products = len(products_list)
        self.alphas = np.array(alphas)
        self.margins = np.array(margins)
        self.verbose = verbose
        lambda_mat = np.full((self.n_products,self.n_products), 0, dtype=float)
        for i in range(0,len(products_list)):
            assert i == products_list[i].getProductNumber()
            endFirst = products_list[i].getSecondaryProduct(0)
            endSecond = products_list[i].getSecondaryProduct(1)
            # if verbose: print("Coordinates {},{} ----- {},{}".format(i,endFirst,i,endSecond))
            lambda_mat[i][endFirst] = 1
            lambda_mat[i][endSecond] = lambda_prob
        #if verbose: print(lambda_mat)
        
        conv_mat = []
        for i in range(0,len(conversion_rates)):
            conv_mat.append(np.full((len(products_list)), conversion_rates[i]).tolist())
        #if verbose: print(conv_mat)
        
        w_matrix = GraphProbabilities(click_prob_matrix)
        lambda_matrix = GraphProbabilities(lambda_mat.tolist())
        conversion_matrix = GraphProbabilities(conv_mat)

        y_matrix = w_matrix.multiplyElementWise(lambda_matrix)
        self.y_matrix = y_matrix.multiplyElementWise(conversion_matrix)
        if verbose: print(self.y_matrix.weightMatrix)

    def computeSingleProduct(self, product):
        firstNode = StepNode(product, [np.array([], dtype=int)], graph_prob=self.y_matrix)
        nodes=[firstNode]
        inverse_prob = np.full((len(self.products_list)), 1)
        # Iterate for #steps times
        for i in range(0, len(self.products_list)-1):
            # Next nodes
            product_nodes = np.full((len(self.products_list)), None)
            for k in range(0,len(nodes)):
                node = nodes[k]
                following = node.expand()
                for j in range(0,len(following)):
                    if following[j].isFeasible() == True:
                        if product_nodes[following[j].product] is not None:
                            product_nodes[following[j].product].merge(following[j])
                        else:
                            product_nodes[following[j].product] = following[j]
            
            # Remove None elements
            existing_nodes = product_nodes[product_nodes != np.array(None)]
            reached_nodes = ""
            for k in range(0,len(existing_nodes)):
                index = existing_nodes[k].product
                reached_nodes = reached_nodes + str(index) + "; "
                # existing_nodes[k].computeProbability() is the probability of visiting "index" in (i+1)-steps
                inverse_prob[index] = inverse_prob[index] * (1 - existing_nodes[k].computeProbability())
            if self.verbose: print("Reached nodes in {}-step: {}".format(i+1,reached_nodes))
            nodes = existing_nodes
        # Probability of visiting product
        return  1 - inverse_prob 

    def computeMargin(self):
        single_margins = np.full((len(self.products_list)), 0)
        for i in range(0,len(self.products_list)):
            visiting_prob = self.computeSingleProduct(i)
            # Margin if alpha = [1 0 0 0 0]
            single_margins[i] = np.multiply(visiting_prob,np.multiply(self.margins,self.conversion_rates)).sum()
            if self.verbose: print("Expected value margin for product {} as starting is {} \n".format(i, single_margins[i]))
        # Weight the single margin by alpha
        return np.multiply(single_margins, self.alphas).sum()
            