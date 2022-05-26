from Model.Evaluator.Evaluator import Evaluator
from Model.GraphProbabilities import GraphProbabilities
from Model.Evaluator.StepNode import StepNode
import numpy as np
from Model.Evaluator.Evaluator import Evaluator

class OneStepEvaluator(Evaluator):
    def __init__(self, products_list=[], click_prob_matrix=None, lambda_prob=0.5, conversion_rates=[], 
                alphas=[], margins=[], units_mean=[], verbose=False):
        super(OneStepEvaluator, self).__init__(products_list=products_list, click_prob_matrix=click_prob_matrix, lambda_prob=lambda_prob, conversion_rates=conversion_rates, 
                alphas=alphas, margins=margins, units_mean=units_mean, verbose=verbose)

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
        
        w_matrix = GraphProbabilities(self.click_prob_matrix)
        lambda_matrix = GraphProbabilities(lambda_mat.tolist())
        conversion_matrix = GraphProbabilities(conv_mat)

        y_matrix = w_matrix.multiplyElementWise(lambda_matrix)
        self.y_matrix = y_matrix.multiplyElementWise(conversion_matrix)
        for i in range(0,len(self.alphas)):
            self.y_matrix.weightMatrix[i][i] = 1
        # print("Y Matrix: {}".format(self.y_matrix.weightMatrix))

        #if verbose: print(self.y_matrix.weightMatrix)

    def computeSingleProduct(self, product):
        y_vector = self.y_matrix.weightMatrix[product]
        profit_vector = np.multiply(np.multiply(y_vector, self.conversion_rates), np.multiply(self.units_mean, self.margins))
        return profit_vector.sum() 

    def computeMargin(self):
        single_margins = np.full((len(self.products_list)), 0)
        for i in range(0,len(self.products_list)):
            single_margins[i] = self.computeSingleProduct(i)
            #if self.verbose: print("Expected value margin for product {} as starting is {} \n".format(i, single_margins[i]))
        # Weight the single margin by alpha
        return np.multiply(single_margins, self.alphas).sum()
            