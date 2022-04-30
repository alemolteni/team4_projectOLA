
class GraphProbabilities:
    """
    GraphProbabilities models the graph that has the five products as nodes and for each edge between nodes P and P' the
    probability that the user clicks on product P' given P as primary
    """
    def __init__(self, weightMatrix):
        """
        Parameters
        ----------    
        weigthMatrix : matrix 
        Shape is #Products (From) x #Products (To) and its (i,j) value states the probability of clicking a product given that it has been seen, i.e. P(click j | seen j, primary i)
        
        E.G. [[0,.54][.12,0]] the probability of going from first product to second (given that is shown) is 0.5
        """
        self.weightMatrix = weightMatrix

    
    def getEdgeProbability(self, startNode, endNode):
        """
        This method returns the click probability of a product P' (endNode) while displayed
        in the FIRST OF THE SECONDARIES SLOT of product P (startNode)
        """
        return self.weightMatrix[startNode][endNode]

    def shape(self):
        return (len(self.weightMatrix),len(self.weightMatrix[0]))

    def multiplyElementWise(self, graph_prob):
        for i in range(0,len(self.weightMatrix)):
            for j in range(0,len(self.weightMatrix[0])):
                self.weightMatrix[i][j] = self.weightMatrix[i][j] * graph_prob.getEdgeProbability(i,j)
        return self