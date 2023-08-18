import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import seaborn as sns



def delete_rows_before_2023(df):
    # because we trained model with 2023 data
    df = df.query("tpep_pickup_datetime.dt.year >= 2023 & tpep_dropoff_datetime.dt.year >= 2023")
    return df

'''
Imputing null values of passenger_count

'''

def set_time_as_index(df):
    df.set_index("tpep_pickup_datetime", inplace=True)
    return df

def impute_passenger_count_10m(x):  # x is dataset for 10-min timestamp
    missed_passengers = x[x.passenger_count.isna()]
    if len(missed_passengers) != 0:
        median_passengers = x.groupby('PULocationID', as_index=False).median()
        joined_x = pd.merge(missed_passengers, median_passengers, on='PULocationID', how='inner')
        x['passenger_count'].loc[x.passenger_count.isna()] = joined_x['passenger_count_y']
        x['passenger_count'].fillna(0, inplace=True)
    return x

def impute_passenger_count(df): # impute null values
    df = df.groupby(pd.Grouper(freq='10T')).apply(impute_passenger_count_10m)
    return df
    





