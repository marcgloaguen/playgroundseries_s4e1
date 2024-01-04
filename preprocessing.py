import pandas as pd



class Preprocessing:
    def __init__(self, train:pd.DataFrame, test:pd.DataFrame, target:str) :
        self.X = train.drop(columns=target)
        self.test = test
        self.target = train[target]

    def __geography(self):
        self.X = pd.get_dummies(self.X, columns=['geography'])
        self.test = pd.get_dummies(self.test, columns=['geography'])

    def __gender(self, feat_fact):
        self.X = pd.factorize(self.X.Gender, sort=True)
        self.test = pd.factorize(self.test.Gender, sort=True)

    def encoded(self):
        self.__geography()
        self.__gender()
        return self.X, self.target, self.test
