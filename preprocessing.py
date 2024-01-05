import pandas as pd
from sklearn.preprocessing import StandardScaler

class Preprocessing:
    def __geography(self):
        self.data = pd.get_dummies(self.data, columns=['Geography'])

    def __gender(self):
        self.data.loc[:, 'Gender'], self.gender_uniques = pd.factorize(self.data.Gender, sort=True)

    def __normal_fit_transform(self):
        self.scaler = StandardScaler()
        col_norm = ['CreditScore', 'Age', 'Balance', 'Tenure', 'NumOfProducts', 'EstimatedSalary']
        self.data.loc[:, col_norm] = self.scaler.fit_transform(self.data[col_norm])

    def __normal_transform(self):
        col_norm = ['CreditScore', 'Age', 'Balance', 'Tenure', 'NumOfProducts', 'EstimatedSalary']
        self.data.loc[:, col_norm] = self.scaler.transform(self.data[col_norm])

    def fit_transform(self, data: pd.DataFrame):
        self.data = data
        self.__geography()
        self.__gender()
        self.__normal_fit_transform()
        return self.data

    def transform(self, data: pd.DataFrame):
        self.data = data
        self.__geography()
        self.__gender()
        self.__normal_transform()
        return self.data
