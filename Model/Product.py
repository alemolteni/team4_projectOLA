# The 'Product' class is used to portray a single product which is characterized by a number
# and an ulterior list attribute containing the hierarchical order of secondary products linked to the product in exam

class Product:
    def __init__(self, productNumber, secondaryProducts):
        self.productNumber = productNumber
        self.secondaryProducts = secondaryProducts

    def getProductNumber(self):
        return self.productNumber

    def getSecondaryProduct(self, secondaryNumber):
        return self.secondaryProducts[secondaryNumber]

    def getSecondaryProductList(self):
        return self.secondaryProducts