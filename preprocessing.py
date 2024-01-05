import pandas as pd
from sklearn.preprocessing import StandardScaler


class Preprocessing:
    def __init__(self, train: pd.DataFrame, test: pd.DataFrame, target: str):
        self.X = train.drop(columns=target)
        self.test = test
        self.target = train[target]

    def __geography(self):
        self.X = pd.get_dummies(self.X, columns=['Geography'])
        self.test = pd.get_dummies(self.test, columns=['Geography'])

    def __gender(self):
        self.X.loc[:, 'Gender'], self.gender_uniques = pd.factorize(
            self.X.Gender, sort=True
            )
        self.test.loc[:, 'Gender'] = pd.factorize(
            self.test.Gender, sort=True
            )[0]

    def _normal(self):
        scaler = StandardScaler()
    
    
    def encoded(self):
        self.__geography()
        self.__gender()
        return self.X, self.target, self.test
