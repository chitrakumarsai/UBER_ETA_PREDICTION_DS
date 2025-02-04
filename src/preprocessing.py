import pickle
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

class DataPreprocessing:
    def __init__(self) -> None:
        """
        Earth's radius in kilometers
        """
        self.R = 6371
    def update_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Update column names
        """
        df.rename(columns = {'Weatherconditions': 'Weather_conditions'}, inplace = True)
        df.columns = df.columns.str.lower().str.replace(' ', '_')
        return df
    
    def extract_feature_value(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract feature value
        - Extract weather condition value from weather conditions
        - Create a new city_code column from he delivery_person_id column
        - Strip leading/trailing whitespaces from the object type columns
        """
        df['weather_conditions'] = df['weather_conditions'].apply(lambda x: x.split(' ')[-1].strip())
        df['city_code'] = df['delivery_person_id'].str.split('RES', expand= True)[0]
        df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
        return df
    
    def extract_label_value(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extracts the label value (Time_taken(min)) from the 'Time_taken(min)' column in the given DataFrame df.
        """
        df['time_taken(min)'] = df['time_taken(min)'].apply(lambda x: int(x.split(' ')[1].strip()))
        return df
       
    def drop_columns(self, df):
        df.drop(['ID', 'Delivery_person_ID'], axis=1, inplace=True)

    def update_datatype(self, df):
        """
        Updates the data types of the following columns in the given DataFrame df:
        - 'delivery_person_Age' to float64
        - 'delivery_person_ratings' to float64
        - 'multiple_deliveries' to float64
        - 'order_date' to datetime with format "%d-%m-%Y"
        """
        
        df['delivery_person_age'] = df['delivery_person_age'].astype('float64')
        df['delivery_person_ratings'] = df['delivery_person_ratings'].astype('float64')
        df['multiple_deliveries'] = df['multiple_deliveries'].astype('float64')
        df['order_date'] = pd.to_datetime(df['order_date'], format="%d-%m-%Y")

    def convert_nan(self, df):        
        """
        Converts the string 'NaN' to a float NaN value in the given DataFrame df.
        """
        
        df.replace('NaN', float(np.nan), regex=True, inplace=True)

    def handle_null_values(self, df):
        """
        Fills null values in 'delivery_person_age' column with the median value of the 'delivery_person_age' column grouped by 'city_code'.
        Fills null values in 'wather_conditions' column with the mode value of the 'weather_conditions' column grouped by 'city_code'.
        Fills null values in 'delivery_person_ratings' with the column median value grouped by 'city_code'.
        Fills null values in 'time_orderd' with the corresponding 'time_order_picked' value.
        Fills null values in 'Road_traffic_density', 'Multiple_deliveries', 'Festival', and 'City_type' with the most frequent value

        """
        df['delivery_person_age'] = df.groupby('city_code')['delivery_person_age'].apply(lambda x: x.fillna(x.median()))
        df['weather_conditions'] = df.groupby('city_code')['weather_conditions'].apply(lambda x: x.fillna(x.mode()))
        df['delivery_person_ratings'] = df.groupby('city_code')['delivery_person_ratings'].apply(lambda x: x.fillna(x.median()))
        df['time_ordered'] = df['time_ordered'].fillna(df['Time_Order_picked'])

        mode_imp = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
        mode_cols = ["road_traffic_density",
                "multiple_deliveries", "festival", "city"]

        for col in mode_cols:
            df[col] = mode_imp.fit_transform(df[col].to_numpy().reshape(-1,1)).ravel()



