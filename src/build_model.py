from preprocessing import *

# Load the data
df_train = pd.read_csv('./data/external/train.csv')

dp = DataPreprocessing()
dp.cleaning_steps(df_train)                                # Perform Cleaning
dp.extract_label_value(df_train)                           # Extract Label Value
dp.perform_feature_engineering(df_train)                   # Perform feature engineering

# Split features & label
X = df_train.drop('time_taken(min)', axis=1)               # Features
y = df_train['time_taken(min)']                            # Target variable

label_encoders = dp.label_encoding(X)                      # Label Encoding
X_train, X_test, y_train, y_test = dp.data_split(X, y)     # Test Train Split
X_train, X_test, scaler = dp.standardize(X_train, X_test)  # Standardization

# Build Model
model = xgb.XGBRegressor(n_estimators=20, max_depth=9)
model.fit(X_train, y_train)

# Evaluate Model
y_pred = model.predict(X_test)
dp.evaluate_model(y_test, y_pred)

# Create model.pkl and Save Model
with open("./models/model.pickle", 'wb') as f:
    pickle.dump((model, label_encoders, scaler), f)
print("Model pickle saved to model folder")
