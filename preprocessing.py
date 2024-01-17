import pandas as pd
from sklearn.preprocessing import StandardScaler, PowerTransformer, MinMaxScaler


class Preprocessing:
    def __geography(self):
        self.data = pd.get_dummies(self.data, columns=['Geography'])

    def __gender(self):
        self.data.loc[:, 'Gender'], self.gender_uniques = pd.factorize(self.data.Gender, sort=True)

    def __normal_fit_transform(self):
        self.scaler = StandardScaler()
        col_norm = ['CreditScore','EstimatedSalary']
        self.data.loc[:, col_norm] = self.scaler.fit_transform(self.data[col_norm])
        
    def __normal_transform(self):
        col_norm = ['CreditScore', 'EstimatedSalary']
        self.data.loc[:, col_norm] = self.scaler.transform(self.data[col_norm])

    def __power_fit_transform(self):
        self.power = PowerTransformer()
        col_norm = ['Age', 'Balance']
        self.data.loc[:, col_norm] = self.power.fit_transform(self.data[col_norm])
    
    def __power_transform(self):
        col_norm = ['Age', 'Balance']
        self.data.loc[:, col_norm] = self.power.transform(self.data[col_norm])
    
    def __MinMax_fit_transform(self):
        self.minmax = MinMaxScaler()
        col_norm = ['Tenure', 'NumOfProducts']
        self.data.loc[:, col_norm] = self.minmax.fit_transform(self.data[col_norm])
    
    def __MinMax_transform(self):
        col_norm = ['Tenure', 'NumOfProducts']
        self.data.loc[:, col_norm] = self.minmax.transform(self.data[col_norm])
    
    
    
    def fit_transform(self, data: pd.DataFrame):
        self.data = data
        self.__geography()
        self.__gender()
        self.__normal_fit_transform()
        self.__power_fit_transform()
        self.__MinMax_fit_transform()
        return self.data

    def transform(self, data: pd.DataFrame):
        self.data = data
        self.__geography()
        self.__gender()
        self.__normal_transform()
        self.__power_transform()
        self.__MinMax_transform()
        return self.data
