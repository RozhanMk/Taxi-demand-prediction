"""
Before running this code, you should impute or delete all null values of passenger counts.
"""

import pandas as pd
#import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
#import seaborn as sns
from datetime import datetime, timedelta
from sklearn.ensemble import IsolationForest
from scipy.stats import zscore

"""
Method 1 : Using Isolation Forest to identify passenger_count outliers and using 
Quantile based flooring and capping to impute outliers
"""
def isolated_forest_and_quantile(df):
    features = ['passenger_count']
    # Create an Isolation Forest model with contamination set to 0.05 (5% of data are expected to be outliers)
    iso_forest = IsolationForest(n_estimators=100, contamination=0.05)
    # Fit the model to the selected features
    iso_forest.fit(df[features])
    # Predict the outliers using the model
    outliers = iso_forest.predict(df[features])
    outlier_mask = outliers == -1
    outliers_data = df[outlier_mask]
    mask = df.isin(outliers_data.to_dict('list')).all(axis=1)

    # flooring and capping
    floor = min(df.loc[~mask, 'passenger_count'].unique())
    cap = max(df.loc[~mask, 'passenger_count'].unique())
    df.loc[mask, 'passenger_count'] = df.loc[mask, 'passenger_count'].clip(lower=floor, upper=cap)
    df.loc[df["passenger_count"] == 0, "passenger_count"] = 1
    return df

"""
Method 2 : Using Z-score to identify outliers and Quantile based flooring and capping to impute outliers
"""

def z_score_and_quantile(df):
    df['zscore'] = zscore(df['passenger_count'])
    outliers_mask = (df['zscore'] > 4) | (df['zscore'] < -4)
    outliers = df.loc[outliers_mask, 'passenger_count']

    floor = min(df.loc[~outliers_mask, 'passenger_count'].unique())
    cap = max(df.loc[~outliers_mask, 'passenger_count'].unique())
    df.loc[outliers_mask, 'passenger_count'] = outliers.clip(lower=floor, upper=cap)
    df.loc[df["passenger_count"] == 0, "passenger_count"] = 1
    return df
