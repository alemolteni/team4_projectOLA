from Model.Evaluator.Evaluator import Evaluator
from Model.GraphProbabilities import GraphProbabilities
from Model.Evaluator.StepNode import StepNode
import numpy as np
from Model.Evaluator.Evaluator import Evaluator

class GraphEvaluator(Evaluator):
    def __init__(self, products_list=[], click_prob_matrix=None, lambda_prob=0.5, conversion_rates=[], 
                alphas=[], margins=[], units_mean=None, convert_units=True, verbose=False):
        super(GraphEvaluator, self).__init__(products_list=products_list, click_prob_matrix=click_prob_matrix, lambda_prob=lambda_prob, conversion_rates=conversion_rates, 
                alphas=alphas, margins=margins, units_mean=units_mean, convert_units=convert_units, verbose=verbose)

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

        #if verbose: print(self.y_matrix.weightMatrix)

    """
    Compute the probability of visiting a node starting from product i
    """
    def computeSingleProduct(self, product):
        firstNode = StepNode(product, [np.array([], dtype=int)], graph_prob=self.y_matrix, 
                            margins=self.margins, units_mean=self.units_mean, 
                            conversion_rates=self.conversion_rates, verbose=self.verbose)
        nodes=[firstNode]
        step_probability = np.zeros((len(self.alphas),len(self.alphas)))
        #joint_prob = np.full((len(self.products_list)), 0).tolist()
        #joint_prob[product] = 1
        step_probability[0][product] = 1
        # Iterate for #steps times
        for i in range(0, len(self.products_list)-1):
            # Next nodes
            product_nodes = np.full((len(self.products_list)), None)
            # if self.verbose: print("Nodes to be expanded: {}".format(len(nodes)))
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
            # if self.verbose: print("Nodes reached from previous step: {} total {}".format(product_nodes, len(existing_nodes)))
            for k in range(0,len(existing_nodes)):
                index = existing_nodes[k].product
                reached_nodes = reached_nodes + str(index) + "; "

                # existing_nodes[k].computeProbability() is the probability of visiting "index" in (i+1)-steps
                reaching_probability = existing_nodes[k].computeProbability()
                step_probability[i+1][index] = reaching_probability

            # if self.verbose: print("Probability of visiting nodes in at most {}-step from {}: {}".format(i+1, product, joint_prob))
            nodes = existing_nodes

        # Probability of visiting product is given by:
        # P(A->B) = P(AB in 1-step) + P(AB in 2-step)*(1-P(AB in 1-step)) ...
        remaining_prob_space = np.full(len(self.alphas), 1)
        cumulative_sum = np.full(len(self.alphas), 0)
        for k in range(0,len(self.alphas)):
            cumulative_sum = np.add(cumulative_sum, step_probability[k] * remaining_prob_space)
            remaining_prob_space = remaining_prob_space * (1 - step_probability[k])

        return cumulative_sum

    """
    Compute the probability of visiting a node starting from product i
    """
    def getRewardSingleProduct(self, product):
        firstNode = StepNode(product, [np.array([], dtype=int)], graph_prob=self.y_matrix, 
                            margins=self.margins, units_mean=self.units_mean, 
                            conversion_rates=self.conversion_rates, verbose=self.verbose)
        nodes=[firstNode]
        step_probability = np.zeros((len(self.alphas),len(self.alphas)))
        step_probability[0][product] = 1

        step_reward = np.zeros((len(self.alphas),len(self.alphas)))
        step_reward[0][product] = self.conversion_rates[product] * self.units_mean[product] * self.margins[product]
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
            # if self.verbose: print("Nodes reached from previous step: {} total {}".format(product_nodes, len(existing_nodes)))
            for k in range(0,len(existing_nodes)):
                index = existing_nodes[k].product
                reached_nodes = reached_nodes + str(index) + "; "

                # existing_nodes[k].computeProbability() is the probability of visiting "index" in (i+1)-steps
                reaching_probability = existing_nodes[k].computeProbability()
                step_probability[i+1][index] = reaching_probability

                expected_reward = existing_nodes[k].computeExpectedReward()
                step_reward[i+1][index] = expected_reward

            # if self.verbose: print("Probability of visiting nodes in at most {}-step from {}: {}".format(i+1, product, joint_prob))
            nodes = existing_nodes

        # Expected reward must be scaled by the probability of a non-visit in earlier steps
        # E[rew from A] = E[rew 0-step from A] + E[rew 1-step from A]*(1-P(i TO A in 0-step)) ...
        remaining_prob_space = np.full(len(self.alphas), 1)
        cumulative_sum = np.full(len(self.alphas), 0)
        for k in range(0,len(self.alphas)):
            cumulative_sum = np.add(cumulative_sum, step_reward[k] * remaining_prob_space)
            remaining_prob_space = remaining_prob_space * (1 - step_probability[k])
        #print(step_reward)
        return cumulative_sum.sum()

    def computeMargin(self):
        #expected_rewards = []
        #for i in range(0,len(self.products_list)):
        #    expected_rew = self.getRewardSingleProduct(i)
        #    expected_rewards.append(expected_rew)

        #return np.multiply(expected_rewards, self.alphas).sum()

        all_visit_prob = self.getVisitingProbability()
        products_profit = all_visit_prob * self.conversion_rates * self.units_mean * self.margins
        #print(products_profit)
        return products_profit.sum()

        single_margins = np.full((len(self.products_list)), 0)
        for i in range(0,len(self.products_list)):
            visiting_prob = self.computeSingleProduct(i)
            if self.verbose: print("Visiting probability from product {}: {}".format(i, visiting_prob))
            if (visiting_prob > np.full(len(visiting_prob),1.1)).any() and False:
                self.verbose = True
                print("====== ASSERT WARNING ! DEBUG MODE =======")
                visiting_prob = self.computeSingleProduct(i)
                print("====== ASSERT WARNING ! DEBUG MODE END =======")
                self.verbose = False
            # assert (visiting_prob <= np.full(len(visiting_prob),1.1)).all(), "Probability of visiting greater than one {}, margins {}".format(visiting_prob, self.margins)
            visiting_prob[visiting_prob > 1] = 1
            # Margin if alpha = [1 0 0 0 0]
            single_margins[i] = np.multiply(visiting_prob,np.multiply(np.multiply(self.margins,self.units_mean),self.conversion_rates)).sum()
            #if self.verbose: print("Expected value margin for product {} as starting is {} \n".format(i, single_margins[i]))
        # Weight the single margin by alpha
        return np.multiply(single_margins, self.alphas).sum()

    def getVisitingProbability(self):
        weighted_visit_prob = np.zeros(len(self.products_list))
        for i in range(0,len(self.products_list)):
            visiting_prob = self.computeSingleProduct(i)
            weighted_visit_prob = weighted_visit_prob + (self.alphas[i] * visiting_prob)
        
        return weighted_visit_prob

            