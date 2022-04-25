from Environment import *
from Model.UserClass import *
from Model.Product import *
from Model.constants import *
from Model.GraphProbabilities import *

# ------> 'Lambda' da inserire in 'constants.py'
Lambda = 0.5 #da cambiare


userClasses = [UserClass(conversionRate=[[.23,.33],[.43,.23],[.10,.5]])] #First kid, second adult man, third adult woman

# Creation of the product list: each element of the list contains a 'Product' type object
# NB: each pair of secondary products is fixed for every product. [see specifics]
productList = [Product(int(key), SECONDARY_PRODUCTS[key]) for key in SECONDARY_PRODUCTS]

# Graph's edges click-probabilities
clickProbability = GraphProbabilities(PROBABILITY_MATRIX)

environment = Environment(Lambda, userClasses)