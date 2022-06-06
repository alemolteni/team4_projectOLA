# Since in every configuration file there are three different possible user's classes, 
# a method to merge the possible different classes' parameters is needed in order to compute 
# a price configuration that maximizes the overall margin for each class of users.
# The method below creates a new configuration by computing the weigthed mean, with respect to the userMean, 
# of all the class parameters. Namely: lambda, alphas, conversionRates, clickProbability, unitsShape.

import numpy as np
import pprint
import json
from Model.Product import *

def mergeUserClasses(PathList, debug):
    mergedConfigurationList = []
    for c_path in PathList:

        if debug: print("#### Megring classes of {} #####".format(c_path))

        lambdaList = []
        alphaList = []
        conversionRateList = []
        clickProbabilityList = []
        unitsShapeList = []
        userMeanList = []
        actualUnitsMeanList = []

        f = open(c_path)
        configuration = json.load(f)
        f.close()
        uc = configuration["classes"][0]

        # Fixed parameters for all classes:
        optimalMargin = configuration["optimalMargin"]
        optimalConfig = configuration["optimalConfig"]
        margins = configuration["margins"]
        productList = [Product(int(key), uc["secondary"][key]) for key in uc["secondary"]]

        count = 0
        for userClass in configuration["classes"]:

            if debug: print("userClass n.", count)
            count += 1

            # To compute the weigthed mean it is useful to convert matrices and lists into numpy.arrays
            lambdaList.append(np.array(userClass["lambda"]))
            alphaList.append(np.array(userClass["alphas"]))
            conversionRateList.append(np.array(userClass["conversionRates"]).tolist())
            clickProbabilityList.append(np.array(userClass["clickProbability"]).tolist())
            unitsShapeList.append(np.array(userClass["unitsShape"]))
            userMeanList.append(userClass["usersMean"])
            actualUnitsMeanList.append(userClass["actualUnitsMean"])
        
        # lambdaList[] ... unitShapeList[] now contain the class parameters for each user in numpy.array form
        # Now compute the weigthed means and create a new configuration

        lambdaNpArray = np.average(lambdaList, weights=userMeanList, axis=0)
        alphasNpArray = np.average(alphaList, weights=userMeanList, axis=0)
        conversionRateNpArray = np.average(conversionRateList, weights=userMeanList, axis=0)
        clickProbabilityNpArray = np.average(clickProbabilityList, weights=userMeanList, axis=0)
        unitsShapeNpArray = np.average(unitsShapeList, weights=userMeanList, axis=0)
        actualUnitsMeanNpArray = np.average(actualUnitsMeanList, weights=userMeanList, axis=0)

        if debug:
            print("UsersMean: ", userMeanList)

            print("\n### AVERAGE CHECK FOR {}: ###".format(c_path))
            print("\nLAMBDA:")
            for i in range(0, len(configuration["classes"])):
                print("lambda[{}]: {}".format(i, lambdaList[i]))
            print("averageLambda: {}".format(lambdaNpArray))

            print("\nALPHAS:")
            for i in range(0, len(configuration["classes"])):
                print("alphaList[{}]: {}".format(i, alphaList[i]))
            print("averageAlphas: {}".format(alphasNpArray))

            print("\nCONV RATES:")
            for i in range(0, len(configuration["classes"])):
                print("conversionRateList[{}]: {}".format(i, conversionRateList[i]))
            print("averageConversionRate: {}".format(conversionRateNpArray))

            print("\nCLICK PROB:")
            for i in range(0, len(configuration["classes"])):
                print("clickProbabilityList[{}]: {}".format(i, clickProbabilityList[i]))
            print("averageClickProbability: {}".format(clickProbabilityNpArray))

            print("\nUSER MEAN:")
            for i in range(0, len(configuration["classes"])):
                print("unitsMeanList[{}]: {}".format(i, unitsShapeList[i]))
            print("averageUnitsMean: {}".format(unitsShapeNpArray))


        # Convert np.arrays back to lists[] and matrices

        Lambda = lambdaNpArray.tolist()
        alphas = alphasNpArray.tolist()
        conversionRates = conversionRateNpArray.tolist()
        clickProbability = clickProbabilityNpArray.tolist()
        unitsShape = unitsShapeNpArray.tolist()
        actualUnitsMean = actualUnitsMeanNpArray.tolist()

        mergedConfiguration = {
            "configurationPath": c_path,
            "productList": productList,
            "optimalMargin": optimalMargin,
            "optimalConfig": optimalConfig,
            "marginsPerPrice": margins, 
            "lambda_p": Lambda,
            "alphas": alphas,
            "conversionRateLevels": conversionRates,
            "click_prob": clickProbability,
            "units_mean": unitsShape,
            "num_prices": len(conversionRates[0]), 
            "num_prods": len(alphas),
            "actual_units_mean": actualUnitsMean
        }

        if debug:
            print("\nMerged configuration:")
            pprint.pprint(mergedConfiguration)

        # finally mergedConfigurationList will contain a merged configuration for each config file.json present in configurationPathList 
        mergedConfigurationList.append(mergedConfiguration)
    return mergedConfigurationList