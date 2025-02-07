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
    def update_column_name(self, df: pd.DataFrame) -> pd.DataFrame:
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
        df['city_code'] = df['delivery_person_id'].str.split('res', expand= True)[0]
        df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
        return df
    
    def extract_label_value(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extracts the label value (Time_taken(min)) from the 'Time_taken(min)' column in the given DataFrame df.
        """
        df['time_taken(min)'] = df['time_taken(min)'].apply(lambda x: int(x.split(' ')[1].strip()))
        return df
       
    def drop_columns(self, df):
        df.drop(['id', 'delivery_person_id'], axis=1, inplace=True)

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
        Handles null values in multiple columns of the DataFrame by imputing appropriate statistics.
        """
        # Fixing groupby().apply() issue with reset_index(drop=True)
        df['delivery_person_age'] = (
            df.groupby('city_code')['delivery_person_age']
            .apply(lambda x: x.fillna(x.median()))
            .reset_index(level=0, drop=True)  # Ensure correct alignment
        )

        df['weather_conditions'] = (
            df.groupby('city_code')['weather_conditions']
            .apply(lambda x: x.fillna(x.mode().iloc[0] if not x.mode().empty else np.nan))
            .reset_index(level=0, drop=True)
        )

        df['delivery_person_ratings'] = (
            df.groupby('city_code')['delivery_person_ratings']
            .apply(lambda x: x.fillna(x.median()))
            .reset_index(level=0, drop=True)
        )

        # Fill null values in 'time_ordered' with 'time_order_picked'
        df['time_ordered'] = df['time_ordered'].fillna(df['time_order_picked'])

        # Handling categorical columns with the most frequent value
        mode_imp = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
        mode_cols = ["road_traffic_density", "multiple_deliveries", "festival", "city"]

        for col in mode_cols:
            df[col] = mode_imp.fit_transform(df[[col]]).ravel()  # Fix column reshaping issue

        return df  # Ensure the function returns the modified DataFrame

    def extract_date_features(self, df):
        """
        Extracts date features from the 'Order_Date' column in the given DataFrame df:
        - 'is_weekend' (boolean): True if the day of the week is Saturday or Sunday
        - 'month_intervals' (categorical): 'start_month' if day <= 10, 'middle_month' if day <= 20, 'end_month' otherwise
        - 'year_quarter' (categorical): The quarter of the year (1, 2, 3, or 4)
        """
        
        df["is_weekend"] = df["order_date"].dt.day_of_week > 4

        df["month_intervals"] = df["order_date"].apply(lambda x: "start_month" if x.day <=10
                                                    else ("middle_month" if x.day <= 20 else "end_month"))

        df["year_quarter"] = df["order_date"].apply(lambda x: x.quarter)

    def calculate_time_diff(self, df):
        """
        Calculates the time difference between order placement and order pickup in the given DataFrame df:
        - Converts 'Time_Ordered' and 'Time_Order_picked' to timedelta
        - Calculates 'Time_Order_picked_formatted' and 'Time_Ordered_formatted' based on 'Order_Date'
        - Calculates 'order_prepare_time' as the difference between 'Time_Order_picked_formatted' and 'Time_Ordered_formatted' in minutes
        - Fills null values in 'order_prepare_time' with the column median
        - Drops 'Time_Ordered', 'Time_Order_picked', 'Time_Ordered_formatted', 'Time_Order_picked_formatted', and 'Order_Date' columns
        """
        
        df['time_ordered'] = pd.to_timedelta(df['time_ordered'])
        df['time_order_picked'] = pd.to_timedelta(df['time_order_picked'])

        df['time_order_picked_formatted'] = df['order_date'] + pd.to_timedelta(np.where(df['time_order_picked'] < df['time_ordered'], 1, 0), unit='D') + df['time_order_picked']
        df['time_ordered_formatted'] = df['order_date'] + df['time_ordered']
        df['order_prepare_time'] = (df['time_order_picked_formatted'] - df['time_ordered_formatted']).dt.total_seconds() / 60

        df['order_prepare_time'].fillna(df['order_prepare_time'].median(), inplace=True)
        df.drop(['time_ordered', 'time_order_picked', 'time_ordered_formatted', 'time_order_picked_formatted', 'order_date'], axis=1, inplace=True)


    def deg_to_rad(self, degrees):
        """
        Converts degrees to radians.
        """
        
        return degrees * (np.pi/180)
    
    def distcalculate(self, lat1, lon1, lat2, lon2):
        """
        Calculates the distance between two latitude-longitude coordinates using the Haversine formula.
        """
        
        lat1, lon1, lat2, lon2 = map(float, [lat1, lon1, lat2, lon2])
        d_lat = self.deg_to_rad(lat2-lat1)
        d_lon = self.deg_to_rad(lon2-lon1)
        a1 = np.sin(d_lat/2)**2 + np.cos(self.deg_to_rad(lat1))
        a2 = np.cos(self.deg_to_rad(lat2)) * np.sin(d_lon/2)**2
        a = a1 * a2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        return self.R * c
    
    def calculate_distance(self, df):
        """
        Calculates the distance between the restaurant and delivery location in the given DataFrame df:
        - Creates a new 'distance' column
        - Calculates the distance using the 'Restaurant_latitude', 'Restaurant_longitude', 'Delivery_location_latitude', and 'Delivery_location_longitude' columns
        - Converts the 'distance' column to int64
        """
        
        df['distance'] = np.nan

        for i in range(len(df)):
            df.loc[i, "distance"] = self.distcalculate(df.loc[i, 'restaurant_latitude'],
                                                    df.loc[i, 'restaurant_longitude'],
                                                    df.loc[i, 'delivery_location_latitude'],
                                                    df.loc[i, 'delivery_location_longitude'])
        df["distance"] = df["distance"].astype("int64")



    def label_encoding(self, df):
        """
        Performs label encoding on categorical columns in the given DataFrame df:
        - Identifies object columns
        - Strips leading/trailing whitespace from object columns
        - Fits and transforms each object column using LabelEncoder
        - Returns a dictionary of LabelEncoder objects for each encoded column
        """
        
        categorical_columns = df.select_dtypes(include='object').columns
        label_encoders = {}

        for column in categorical_columns:
            df[column] = df[column].str.strip()
            label_encoder = LabelEncoder()
            label_encoder.fit(df[column])
            df[column] = label_encoder.transform(df[column])
            label_encoders[column] = label_encoder
        return label_encoders
    
    def data_split(self, X, y):        
        """
        Splits the input features X and target variable y into training and testing sets:
        - Splits the data with a test size of 0.2 and a random state of 42
        - Returns X_train, X_test, y_train, y_test
        """
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test
    
    def standardize(self, X_train, X_test):
        """
        Standardizes the training and testing feature sets:
        - Fits a StandardScaler on X_train
        - Transforms X_train and X_test using the fitted StandardScaler
        - Returns X_train, X_test, and the fitted StandardScaler
        """
        
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
        return X_train, X_test, scaler
    

    def cleaning_steps(self, df):
        self.update_column_name(df)
        self.extract_feature_value(df)
        self.drop_columns(df)
        self.update_datatype(df)
        self.convert_nan(df)
        print(df.columns)
        self.handle_null_values(df)

    def perform_feature_engineering(self, df):
        self.extract_date_features(df)
        self.calculate_time_diff(df)
        self.calculate_distance(df)


    def evaluate_model(self, y_test, y_pred):
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)


        print("Mean Absolute Error (MAE):", round(mae, 2))
        print("Mean Squared Error (MSE):", round(mse, 2))
        print("Root Mean Squared Error (RMSE):", round(rmse, 2))
        print("R-squared (R2) Score:", round(r2, 2))
