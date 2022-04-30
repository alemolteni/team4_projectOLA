# Number of product types in the shop
PRODUCT_NUMBER = 5

# Dictionary containing the secondary products pairs linked to each product
SECONDARY_PRODUCTS = {'0': [1, 2], '1': [3, 4], '2': [1, 0], '3': [0, 4], '4': [3, 2]}

# Graph's edges matrix

# PROBABILITY_MATRIX = [[0, 0, 0.2, 0, 0],
#                      [0.1, 0, 0, 0.3, 0],
#                      [0, 0.2, 0, 0.1, 0],
#                      [0.2, 0, 0, 0, 0],
#                      [0, 0, 0, 0, 0]]

PROBABILITY_MATRIX = [[0, 0, 0.9, 0, 0],
                      [0.8, 0, 0, 0.9, 0],
                      [0, 1, 0, 0.7, 0],
                      [0.9, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0]]

PROBABILITY_MATRIX_1 = [[0, 0, 0.2, 0, 0],
                      [0.1, 0, 0, 0.3, 0],
                      [0, 0.2, 0, 0.1, 0],
                      [0.2, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0]]