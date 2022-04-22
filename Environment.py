import numpy as np
from Model.UserClass import *


class Environment:
    def __init__(self, Lambda, classes):
        self.Lambda = Lambda
        self.classes = classes

    def round(self, pulled_arms):

        for userClass in self.classes:
            dailyUser = np.random.normal(userClass.n_user[0], userClass.n_user[1])

        #To be completed