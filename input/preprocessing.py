import pandas as pd
from configparser import ConfigParser
from sklearn.preprocessing import StandardScaler , MinMaxScaler
import numpy as np
import joblib
import os

const_variables = ConfigParser()
const_variables.read("./config/config.ini") 
class load_transform():

    def __init__(self ) -> None:
        pass

    @staticmethod
    def log_transform(data):
        return data.apply(lambda x : np.log(x) , axis = 0)
        
    @staticmethod
    def load()-> pd.DataFrame:
        """
         Load data 
        """
        data_path = const_variables.get("data_path" , "file_path")
        data = pd.read_csv(filepath_or_buffer= data_path , sep= ";")
       
        data.drop("Dates",axis=1,inplace=True)
        data.drop("Unnamed: 0",axis=1,inplace=True)
        
        return data
        

    def mm_scaling(self , withlogtrans = False) -> pd.DataFrame:
        """
        Normalization
        """

        data = self.load()
       
        
        train_split = float(const_variables.get("slicing" , "train_split"))
        n_train = int(train_split * len(data))
        n_test = len(data) - n_train

        if withlogtrans == False:
            features = data.columns
            feature_array = data.values
            # Fit Scaler only on Training features
            feature_scaler = MinMaxScaler()
            feature_scaler.fit(feature_array[:n_train])
            # Fit Scaler only on Training target values
            target_scaler = MinMaxScaler()
            joblib.dump(target_scaler , os.path.abspath("output/normalization.save"))
            target_scaler.fit(feature_array[:n_train, -1].reshape(-1, 1))


            # Transfom on both Training and Test data
            scaled_array = pd.DataFrame(feature_scaler.transform(feature_array),
                                        columns=features)

            return scaled_array
        else:
            log_data = self.log_transform(data)

            features = log_data.columns
            feature_array = log_data.values
            # Fit Scaler only on Training features
            feature_scaler = MinMaxScaler()
            feature_scaler.fit(feature_array[:n_train])
            # Fit Scaler only on Training target values
            target_scaler = MinMaxScaler()
            target_scaler.fit(feature_array[:n_train, -1].reshape(-1, 1))
            joblib.dump(target_scaler , os.path.abspath("src/output/normalization.save"))
            # Transfom on both Training and Test data
            scaled_array = pd.DataFrame(feature_scaler.transform(feature_array),
                                        columns=features)

            return scaled_array

    def std_scaling(self , withlogtrans = False):
        """
        Normalization
        """
        data = self.load()

        train_split = float(const_variables.get("slicing" , "train_split"))
        n_train = int(train_split * len(data))
        n_test = len(data) - n_train

        if withlogtrans == False:
            features = data.columns
            feature_array = data.values
            # Fit Scaler only on Training features
            feature_scaler = StandardScaler()
            feature_scaler.fit(feature_array[:n_train])
            # Fit Scaler only on Training target values
            target_scaler =  StandardScaler()
            target_scaler.fit(feature_array[:n_train, -1].reshape(-1, 1))
            joblib.dump(target_scaler , os.path.abspath("src/output/normalization.save"))
            # Transfom on both Training and Test data
            scaled_array = pd.DataFrame(feature_scaler.transform(feature_array),
                                        columns=features)

            return scaled_array
        else:

            log_data = self.log_transform(data)

            features = log_data.columns
            feature_array = log_data.values
            # Fit Scaler only on Training features
            feature_scaler =  StandardScaler()
            feature_scaler.fit(feature_array[:n_train])
            # Fit Scaler only on Training target values
            target_scaler =  StandardScaler()
            target_scaler.fit(feature_array[:n_train, -1].reshape(-1, 1))
            joblib.dump(target_scaler , os.path.abspath("src/output/normalization.save"))
            # Transfom on both Training and Test data
            scaled_array = pd.DataFrame(feature_scaler.transform(feature_array),
                                        columns=features)

            return scaled_array

    
    @staticmethod
    def create_sliding_window(data):
        sequence_length =  int(const_variables.get("slicing" , "sequence_length"))
        output_length =  int(const_variables.get("slicing" , "output_length"))
        stride=1
        X_list, y_list = [], []
        for i in range(len(data)):
            if (i + sequence_length) < len(data):
                X_list.append(data.iloc[i:i+sequence_length:stride, :].values)
                y_list.append(data.iloc[i:i+output_length, -1])
            return np.array(X_list), np.array(y_list)


    def fit_transform(self, withlogtrans = False , method_scaling="minmax"):
        data = self.load()
        train_split = float(const_variables.get("slicing" , "train_split"))
        n_train = int(train_split * len(data))
        n_test = len(data) - n_train

        if method_scaling=="minmax":
            
            scaled_array =  self.mm_scaling(withlogtrans) 
            X ,Y=self.create_sliding_window(scaled_array )
            X_train = X[:n_train]
            y_train = Y[:n_train]
            X_test = X[n_train:]
            y_test = Y[n_train:]
            return X_train,y_train,X_test,y_test
        else:
            scaled_array =  self.std_scaling(withlogtrans) 
            X,Y =self.create_sliding_window(scaled_array )
            X_train = X[:n_train]
            y_train = Y[:n_train]
            X_test = X[n_train:]
            y_test = Y[n_train:]
            return X_train,y_train,X_test,y_test
        