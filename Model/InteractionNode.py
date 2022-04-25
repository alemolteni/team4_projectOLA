# An 'interactionNode' is generated for each primary product that the user interacts with. At the end of the 'user
# episode' a list of interaction nodes is recursively generated in order to keep track of the actions performed by
# the user on the site

class InteractionNode:

    # Each interaction node keeps track of:
    #   -product: id of the product type involved in the interaction;
    #   -price: price of the product;
    #   -bought: equals to '1' if the product has been bought, '0' otherwise;
    #   -units: units bought;
    #   -following: list of the previous interaction nodes
    def __init__(self, product, price, bought, units, following):
        self.product = product
        self.price = price
        self.bought = bought
        self.units = units
        self.following = following
