import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from . import missings
from . import Outliers_passengers
"""
we will impute nan and outlier values of passenger_count in this file.
"""

class Preprocessing:
    def __init__(self, df, verbose=True):
        self.df = df
        self.verbose = verbose
    
    def fit_transform(self):
        self.run_missing_passenger_and_unimportant()
        self.run_outliers()
        self.df = self.df.reset_index()
        return self.df
    

    def run_missing_passenger_and_unimportant(self):
        if self.verbose == True: print("Imputing missing panssengers")

        self.df = missings.delete_rows_before_2023(self.df)
        self.df = missings.set_time_as_index(self.df)
        self.df = missings.impute_passenger_count(self.df)


    def run_outliers(self):
        if self.verbose == True: print("Imputing outlier passengers")
        self.df = Outliers_passengers.isolated_forest_and_quantile(self.df)









