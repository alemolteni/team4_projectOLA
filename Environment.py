import numpy as np
from Model.UserClass import *
from Model.Product import *
from Model.GraphProbabilities import *

import json

class Environment:
    """
    The enviroment models the behaviour of the project website, it gives at each round a set of interactions/sessions
    """
    def __init__(self, classes=[], config_path=None):
        """
        Initialize the enviroment with the vector of UserClass objects
        """
        #self.classes = classes
        self.classes = list()
        assert len(classes) == 0
        self.t = 0
        self.listener = []
        self.listener_timing = np.array([], dtype=int)
        self.userClassChanges = []
        if config_path != None:
            f = open(config_path)
            config = json.load(f)
            f.close()
            conf_classes = config["classes"]
            for uc in conf_classes:
                productList = [Product(int(key), uc["secondary"][key]) for key in uc["secondary"]]
                if uc["features"][0]["probability"] != 1 and uc["features"][0]["probability"] != 0:
                    # Split the class on the first feature
                    weights = uc["features"][0]["probability"]
                    uc["features"][0]["probability"] = 1
                    self.classes.append(UserClass(conversionRate=uc["conversionRates"],
                                                  clickProbability=uc["clickProbability"],
                                                  alphas=uc["alphas"],
                                                  Lambda=uc["lambda"], n_user_mean=uc["usersMean"]*weights,
                                                  n_user_variance=uc["usersVariance"], productList=productList,
                                                  features_generator=uc["features"],
                                                  units_gamma_shape=uc["unitsShape"]))

                    uc["features"][0]["probability"] = 0
                    self.classes.append(UserClass(conversionRate=uc["conversionRates"],
                                                  clickProbability=uc["clickProbability"],
                                                  alphas=uc["alphas"],
                                                  Lambda=uc["lambda"], n_user_mean=uc["usersMean"]*(1-weights),
                                                  n_user_variance=uc["usersVariance"], productList=productList,
                                                  features_generator=uc["features"],
                                                  units_gamma_shape=uc["unitsShape"]))
                    if "change" in uc:
                        self.userClassChanges.append(uc["change"])
                        self.userClassChanges.append(uc["change"])
                    else:
                        self.userClassChanges.append({"step": -1})
                        self.userClassChanges.append({"step": -1})

                elif uc["features"][1]["probability"] != 1 and uc["features"][1]["probability"] != 0:
                    # Split the class on the second feature
                    weights = uc["features"][1]["probability"]
                    uc["features"][1]["probability"] = 1
                    self.classes.append(UserClass(conversionRate=uc["conversionRates"],
                                                  clickProbability=uc["clickProbability"],
                                                  alphas=uc["alphas"],
                                                  Lambda=uc["lambda"], n_user_mean=uc["usersMean"] * weights,
                                                  n_user_variance=uc["usersVariance"], productList=productList,
                                                  features_generator=uc["features"],
                                                  units_gamma_shape=uc["unitsShape"]))

                    uc["features"][1]["probability"] = 0
                    self.classes.append(UserClass(conversionRate=uc["conversionRates"],
                                                  clickProbability=uc["clickProbability"],
                                                  alphas=uc["alphas"],
                                                  Lambda=uc["lambda"], n_user_mean=uc["usersMean"] * (1 - weights),
                                                  n_user_variance=uc["usersVariance"], productList=productList,
                                                  features_generator=uc["features"],
                                                  units_gamma_shape=uc["unitsShape"]))
                    if "change" in uc:
                        self.userClassChanges.append(uc["change"])
                        self.userClassChanges.append(uc["change"])
                    else:
                        self.userClassChanges.append({"step": -1})
                        self.userClassChanges.append({"step": -1})

                else:
                    self.classes.append(UserClass(conversionRate=uc["conversionRates"],
                                                  clickProbability=uc["clickProbability"],
                                                  alphas=uc["alphas"],
                                                  Lambda=uc["lambda"], n_user_mean=uc["usersMean"],
                                                  n_user_variance=uc["usersVariance"], productList=productList,
                                                  features_generator=uc["features"],
                                                  units_gamma_shape=uc["unitsShape"]))
                    if "change" in uc:
                        self.userClassChanges.append(uc["change"])
                    else:
                        self.userClassChanges.append({"step": -1})

        assert len(self.classes) > 0
        self.n_product = len(self.classes[0].alphas)
        self.price_levels = np.full((self.n_product), 1, dtype=int)


    def addTimeListener(self, fireAt, fireFunction):
        """
        Fire a custom function after "fireAt" rounds (fireAt is a non-negative integer)
        """
        self.listener.append({"id": len(self.listener), "fireAt": fireAt, "fireFunction": fireFunction})
        self.listener_timing = np.append(self.listener_timing, fireAt)

    def setPriceLevels(self, price_levels):
        """
        Set price levels for each product, it must be in the range of feasible levels and its length must be equal to #products
        """
        self.price_levels = price_levels
        for userClass in self.classes:
            userClass.setCurrentPrice(price_levels)

    def setPriceLevelsForContexts(self, price_levels_for_features):
        if len(price_levels_for_features) == 1:
            self.setPriceLevels(price_levels_for_features[0][0])
        else:
            for i in range(0, len(price_levels_for_features)):
                split_feature = price_levels_for_features[i][1]
                for uc in self.classes:
                    compliant = True
                    for f in split_feature:
                        for j in range(0, len(uc.features_names)):
                            if uc.features_names[j] == f:
                                idx = j
                        if uc.features_values[idx] != split_feature[f]:
                            compliant = False
                    if compliant:
                        uc.setCurrentPrice(price_levels_for_features[i][0])

    def round(self):
        """
        Generate a new set of data for the current days, according to the arrival rate of the classes.
        It returns an array of InteractionNode objects
        """
        self.t += 1

        for i in range(0, len(self.classes)):
            if self.userClassChanges[i]["step"] == self.t:
                self.classes[i].conversionRate = self.userClassChanges[i]["conversionRates"]

        # When t reaches a certain value then fire all listener that specified fireAt == self.t
        listener_indexes = np.where(self.listener_timing == self.t)
        listener_indexes = listener_indexes[0]
        for i in range(0, len(listener_indexes)):
            index = listener_indexes[i]
            func = self.listener[index]["fireFunction"]
            func(self.t, self)

        episodes = []
        for userClass in self.classes:
            dailyUsers = math.ceil(np.random.normal(userClass.n_user[0], userClass.n_user[1]))
            userClass.generateNewAlphas()
            for i in range(0,dailyUsers):
                ep = userClass.generateEpisode()
                ep.price_levels = self.price_levels
                episodes.append(ep)
        return  episodes