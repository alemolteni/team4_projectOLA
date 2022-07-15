from Model.Product import Product
from Model.constants import *

productList = [Product(int(key), SECONDARY_PRODUCTS[key]) for key in SECONDARY_PRODUCTS]

for i in range(0, len(productList)):
    print('#Product: ', productList[i].getProductNumber(), 'Secondary: ', productList[i].getSecondaryProductList())
    print('#Product: ', productList[i].getProductNumber(), 'First Secondary: ', productList[i].getSecondaryProduct(0))
    print('#Product: ', productList[i].getProductNumber(), 'Second Secondary: ', productList[i].getSecondaryProduct(1), '\n')

