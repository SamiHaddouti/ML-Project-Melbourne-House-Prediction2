import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler 
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge

from hyperopt import fmin, tpe, hp, STATUS_OK

import mlflow
import mlflow.sklearn
mlflow.set_experiment('DataExploration_Project - Regression')

url = 'https://raw.githubusercontent.com/SamiHaddouti/Machine-Learning-Project/main/data/melb_data.csv'
melb_df = pd.read_csv(url)

# Rename columns
melb_df.rename({'Bedroom2': 'Bedroom', 'Lattitude': 'Latitude', 'Longtitude': 'Longitude'}, axis=1, inplace=True)

# Only keep relevant features (that show correlation with price)
filtered_df2 = melb_df[['Price', 'Rooms', 'YearBuilt', 'Bathroom', 'Car', 'Distance', 'Latitude', 'Longitude']]

# Drop Na(N)
cleaned_df = filtered_df2.dropna()

X = cleaned_df.drop(columns='Price')
y = cleaned_df['Price']

# 70 train 20 val 10 test
# no stratify as this is no classification problem/shuffle to ensure randomized data dsitribution
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.1, random_state=0, shuffle=True)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.15, random_state=0)

print(len(X_train))
print(len(X_val))
print(len(X_test))

# Scaling X
scaler = MinMaxScaler()
  
#X_train = scaler.fit_transform(X_train)  
#X_val = scaler.fit_transform(X_val) 

def eval_model(y_val, y_pred):

    mae = mean_absolute_error(y_val, y_pred)
    mse = mean_squared_error(y_val, y_pred)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    r2 = r2_score(y_val, y_pred)

    metrics = {
        "mae" : mae,
        "mse" : mse,
        "rmse" : rmse,
        "r2" : r2
    }
    
    return metrics

param_space = {
    'alpha': hp.choice('alpha', [0.0001, 0.001, 0.01])
}


def train_model(params):
    # Create and train model.
    
    #ridge_model = KernelRidge()
    ridge_model = Ridge()
    ridge_model.fit(X_train, y_train)

    # Evaluate Metrics
    y_pred = ridge_model.predict(X_val)
    metrics = eval_model(y_val, y_pred)
    mae = mean_absolute_error(y_val, y_pred)

    # Log params, metrics and model to MLflow
    mlflow.log_params(params)
    mlflow.log_metrics(metrics)
    
    mlflow.sklearn.log_model(ridge_model, "ridge_model")
    mlflow.end_run()
    return {"loss": mae, "status": STATUS_OK}


with mlflow.start_run() as run:
    best_params = fmin(
        fn=train_model, 
        space=param_space, 
        algo=tpe.suggest, # Tree of Parzen Estimator instead of random
        max_evals=64)