from Environment import *
from Model.UserClass import *


Lambda = 0.5 #da cambiare
userClasses = [UserClass(), UserClass(), UserClass()] #First kid, second adult man, third adult woman


environment = Environment(Lambda, userClasses)