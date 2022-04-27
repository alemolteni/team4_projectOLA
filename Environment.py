import numpy as np
from Model.UserClass import *

class Environment:
    """
    The enviroment models the behaviour of the project website, it gives at each round a set of interactions/sessions
    """
    def __init__(self, classes):
        """
        Initialize the enviroment with the vector of UserClass objects
        """
        self.classes = classes
        assert len(classes) > 0
        self.n_product = len(classes[0].alphas)
        self.price_levels = np.full((self.n_product), 1, dtype=int)
        self.t = 0
        self.listener = []
        self.listener_timing = np.array([], dtype=int)

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
            
    def round(self):
        """
        Generate a new set of data for the current days, according to the arrival rate of the classes.
        It returns an array of InteractionNode objects
        """
        self.t += 1
        # When t reaches a certain value then fire all listener that specified fireAt == self.t
        listener_indexes = np.where(self.listener_timing == self.t)
        listener_indexes = listener_indexes[0]
        for i in range(0, len(listener_indexes)):
            index = listener_indexes[i]
            func = self.listener[index]["fireFunction"]
            func(self.t, self)

        episodes = []
        for userClass in self.classes:
            dailyUser = math.ceil(np.random.normal(userClass.n_user[0], userClass.n_user[1]))
            for i in range(0,dailyUser):
                episodes.append(userClass.generateEpisode())
        return episodes