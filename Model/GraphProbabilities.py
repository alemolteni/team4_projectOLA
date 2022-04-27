# GraphProbabilities models the graph that has the five products as nodes and for each edge between nodes P and P' the
# probability that the user clicks on product P' given P as primary

class GraphProbabilities:

    def __init__(self, weightMatrix):
        self.weightMatrix = weightMatrix

    # This method returns the click probability of a product P' (endNode) while displayed
    # in the FIRST OF THE SECONDARIES SLOT of product P (startNode)
    # NB: The '-1' accounts for the indices of the matrix starting from [0][0] and finishing in [4][4]
    def getEdgeProbability(self, startNode, endNode):
        return self.weightMatrix[startNode][endNode]