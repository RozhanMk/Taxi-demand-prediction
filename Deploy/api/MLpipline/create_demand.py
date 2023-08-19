import pandas as pd
import geopandas as gpd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import FunctionTransformer
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
import pickle
from .weather import weather
import os
from matplotlib.patches import Patch
from tensorflow.keras.models import load_model as tf_load_model
from api.MLpipline.CleanDatasetPackage import data_preprocessing

CONFIG = {
    "xgboost_model_dir": "api/models/xgboost_model_org_demand_3.sav",
    "date_limit": datetime.strptime('2023-04-30 00:00:00', '%Y-%m-%d %H:%M:%S'),
    "real_demand_dir": "api/MLpipline/evaluate/org_demand_3h.parquet",
    "number_of_zones": 263,
    "target_encoded_zones_dir": "api/utils/mean_demand_per_location.pkl",
    "shapefile_dir": "api/utils/nycZones/taxi_zones.shp",
    "low_demand_limit": 2,
    "high_demand_limit": 41,

}

# main class to do prediction by input dataframe
class Prediction:
    def __init__(self, df_input, name_of_model, timestamp=3, iteration=1):
        self.df_input = df_input
        self.name_of_model = name_of_model
        self.timestamp = timestamp
        self.iteration = int(iteration)

    def predict_xgboost(self):
        # convert input df to demand df(a dataframe that consists of pulocationids and timestamp and demand count)
        df_demand = DemandInputDF(self.df_input, self.timestamp).convert_to_demand_dataset()
        df_demand["timestamp"] = pd.to_datetime(df_demand["timestamp"])

        # history is records of the whole predictions
        history = pd.DataFrame(columns=["timestamp", "PULocationID", "demand_prediction"])
        for i in range(self.iteration):
            if i != 0:
                df_demand = pd.concat([df_demand, predicts], axis=0).reset_index(drop=True)  # add previous predicts to the next iteration.

            # the timestamp that we wanna predict
            needed_datetime = df_demand.timestamp.max() + pd.Timedelta(hours=int(self.timestamp))  
            # get the prediction for one iteration 
            predicts = self.predict_one_iteration(needed_datetime, df_demand)
            predicts["PULocationID"] = range(1, CONFIG.get("number_of_zones") + 1)
            predicts = predicts.reset_index()
            predicts = predicts[["timestamp", "PULocationID", "demand_prediction"]]
            history = history.append(predicts, ignore_index=True)
            predicts = predicts.rename(columns={"demand_prediction" : "demand"})

        history = history.set_index("timestamp")
        # save our predictions to csv file
        history.to_csv("xgboost_predictions.csv")


        # add evaluation if user wanna predict before date limit
        if history.index.max() < CONFIG.get("date_limit"):
            self.plot_prediction(history)
            evaluation = self.evaluate(history)
            return history, evaluation
        else:
            self.plot_prediction(history)
            return history, None



    def predict_one_iteration(self, needed_datetime, df_demand):
        # add features to the df that is gonna be used for prediction
        df_features = PredictionDF(needed_datetime, df_demand, self.timestamp).add_all_features()

        loaded_model = pickle.load(open(CONFIG.get("xgboost_model_dir"), 'rb')) 

        # add predictions column from predicted values
        df_features["demand_prediction"] = loaded_model.predict(df_features)
        # we cannot have negative demand prediction
        df_features["demand_prediction"] = np.where(df_features["demand_prediction"] < 0, 0, df_features["demand_prediction"])
        
        return df_features
    
    
    """
    This will seperate low, medium and high demands on New York map. get the shape file from TLC nyc website.
    """
    def plot_prediction(self, df_features):
        zones_df = gpd.read_file(CONFIG.get("shapefile_dir"))
        df_features['demand_level'] = pd.cut(df_features['demand_prediction'], bins=[0, CONFIG.get("low_demand_limit") + 1, CONFIG.get("high_demand_limit") + 1, float('inf')], labels=["low", "medium" , "high"], right=False)
        df_levels = df_features[["PULocationID", "demand_level"]]
        df_levels = df_levels.rename(columns={"PULocationID": "LocationID"})

        zones_with_level_df = zones_df.merge(df_levels, left_on='LocationID', right_on='LocationID', how='left')

        # Add demand level to the map
        fig = plt.figure(facecolor='#D6D6D6', figsize=(20, 20))
        ax = plt.axes(facecolor='#D6D6D6')      # Plotting
        color_mapping = {
            'low': '#333533',
            'medium': '#FFEE32',
            'high': '#FFD100',
        }
        zones_with_level_df.plot(ax=ax, column='demand_level', linewidth=0.8, edgecolor='black',
                                 color=[color_mapping.get(x) for x in zones_with_level_df['demand_level']], legend=True)

        # Add LocationID labels to the map
        for x, y, label in zip(zones_with_level_df.geometry.centroid.x, zones_with_level_df.geometry.centroid.y, zones_with_level_df['LocationID']):
            ax.text(x, y, label, fontsize=8, ha='center', va='center')

        ax.set_title(f"New York zones based on demand from {df_features.index.min()} to {df_features.index.max()}")
        ax.set_axis_off()
        # Create a custom legend
        legend_elements = [Patch(facecolor=color_mapping[level], edgecolor='black', label=level) for level in zones_with_level_df['demand_level'].unique()]
        ax.legend(handles=legend_elements, loc='lower right')
        plt.savefig('xgboost_demand.png')



    def evaluate(self, df_features):
        real_demand_df = pd.read_parquet(CONFIG.get("real_demand_dir"))
        real_demand_df = real_demand_df.sort_values(["timestamp", "PULocationID"])
        real_demand_df = real_demand_df[["timestamp", "PULocationID", "demand"]]
        real_demand_df = real_demand_df.query(f"timestamp >= '{df_features.index.min()}' & timestamp <= '{df_features.index.max()}'")

        df_features = pd.merge(real_demand_df, df_features, on="PULocationID")

        df_features['demand_level'] = pd.cut(df_features['demand_prediction'], bins=[0, CONFIG.get("low_demand_limit") + 1, CONFIG.get("high_demand_limit") + 1, float('inf')], labels=["low", "medium" , "high"], right=False)

        # use rmse for low demand level
        low_mask = df_features['demand_level'] == 'low'
        low_demand_prediction = df_features.loc[low_mask, 'demand_prediction'].values
        low_demand_real = df_features.loc[low_mask, 'demand'].values

        # use mape for medium and high demand level
        high_mask = df_features['demand_level'].isin(["medium", "high"])
        high_demand_prediction = df_features.loc[high_mask, 'demand_prediction'].values
        high_demand_real = df_features.loc[high_mask, 'demand'].values


        return f"rmse: {np.sqrt(mean_squared_error(low_demand_real, low_demand_prediction))}, mape: {mean_absolute_percentage_error(high_demand_real, high_demand_prediction)}"

    """
    forecast demand with deep model.
    """
    def forecast_deep(self):
        models = self.load_deep_model()
    
        # Select the first model from the list
        selected_model = models[0]

        # Check if the selected model exists
        assert selected_model is not None, f"Model '{self.name_of_model}' not found"

        history = []  # List to store prediction history
        
        fe = FeatureExtractor(self.df_input, self.timestamp)  # Create a FeatureExtractor instance

        for i in range(self.iteration):
            if i == 0:
                df = fe.extract()  # Extract features from the initial batch
            else:
                df = fe.extract(predicts)  # Extract features using previous predictions
            
            date = df.index.max()  # Get the maximum date from the DataFrame
            batch_array = df.to_numpy().astype('float32')  # Convert DataFrame to a float32 numpy array
            predicts = selected_model.predict(batch_array).flatten()  # Make predictions using the selected model
            history.append({
                'date': date.strftime('%Y-%m-%dT%H:%M:%S'),
                'predicts': predicts
            })  # Store current prediction and corresponding date in history
        
        return history

    def load_deep_model(self, load_history=False):
        """
        Load a trained Keras model and optionally its training history.

        Args:
            model_name (str): Name of the model to be loaded.
            load_history (bool, optional): Whether to load training history. Default is False.

        Returns:
            model (tensorflow.keras.Model or None): Loaded Keras model. None if model file doesn't exist.
            history (dict or None): Loaded training history if 'load_history' is True, otherwise None.
        """
        model = None
        history = None

        # Construct paths for the model and history files
        model_dir = f'api/models/{self.name_of_model}'
        model_path = os.path.join(model_dir, f'{self.name_of_model}.keras')
        history_path = os.path.join(model_dir, f'{self.name_of_model}_history.pkl')

        # Load the model if it exists
        if os.path.exists(model_path):
            model = tf_load_model(model_path)

            # Load training history if requested
            if load_history:
                with open(history_path, 'rb') as file:
                    history = pickle.load(file)

        return model, history

# Prepare the raw data set to feed into the model
class FeatureExtractor:
    def __init__(self, df_input, timestamp=1):
        self.df_input = df_input
        self.timestamp = timestamp
        self.df_predictions = pd.DataFrame(columns=['timestamp', 'PULocationID', 'demand'])
        
        # convert input df to demand df(a dataframe that consists of pulocationids and timestamp and demand count)
        self.df_input = DemandInputDF(self.df_input, self.timestamp).convert_to_demand_dataset()
        self.df_input["timestamp"] = pd.to_datetime(self.df_input["timestamp"])



    def extract(self, predicts=None, ):
        current_datetime = self.df_input["timestamp"].max()
        if predicts is not None:
            current_datetime = current_datetime+ pd.Timedelta(hours=self.timestamp)
            for i in range(1,264):
                record = {
                    'timestamp':current_datetime,
                    'PULocationID':i,
                    'demand':predicts[i-1],
                }
                self.df_input.loc[len(self.df_input)] = record

        df_demand = self.df_input.copy()
        needed_datetime = current_datetime+ pd.Timedelta(hours=self.timestamp)
        # add features to the df that is gonna be used for prediction
        df_features = PredictionDF(needed_datetime, df_demand, self.timestamp, deep=True).add_all_features()
        
        return df_features 



        



# get the df from user and create a demand df(a dataframe that consists of pulocationids and timestamp and demand count)
#  --> it will get used for rolling features
class DemandInputDF:
    def __init__(self, df_input, timestamp=3):
        self.timestamp = timestamp
        self.df_input = df_input  # df received from user

    """
    calculate demand for each time interval and LocationID based on our definition of demand.
    demand = Passenger count for current time + (dropoff count for last 6 hours in each LocationID - pickup count for last 6 hours in each LocationID)
    """
    def demand_calculation(self, x, pickups_6h_df, dropoffs_6h_df):
        passenger_sum = x['passenger_count'].sum()
        demand = passenger_sum + 1
        if len(x['tpep_pickup_datetime']) != 0:
            first_time = x['tpep_pickup_datetime'].iloc[0]
            location = x['PULocationID'].iloc[0]
            pickups_6h_def_filtered = pickups_6h_df.loc[(pickups_6h_df.index < first_time) & (pickups_6h_df['PULocationID'] == location)]
            dropoffs_6h_df_filtered = dropoffs_6h_df.loc[(dropoffs_6h_df.index < first_time) & (dropoffs_6h_df['DOLocationID'] == location)]

            if len(pickups_6h_def_filtered) > 0 and len(dropoffs_6h_df_filtered) > 0:
                diff = dropoffs_6h_df_filtered.iloc[-1]['tpep_pickup_datetime'] - pickups_6h_def_filtered.iloc[-1]['tpep_dropoff_datetime']

                if diff > 0 :
                    return demand + diff

        return demand

    def number_of_pickups_in6hour(self, df):
        pickups_6h = df.set_index(df['tpep_pickup_datetime'])
        # Resample the DataFrame to 6-hour intervals and count the number of pickups
        pickups_6h = pickups_6h.groupby([pd.Grouper(freq='6H'), 'PULocationID']).count()

        return pickups_6h


    def number_of_dropoffs_in6hour(self, df):
        dropoffs_6h = df.set_index(df['tpep_dropoff_datetime'])
        # Resample the DataFrame to 6-hour intervals and count the number of drop-offs
        dropoffs_6h = dropoffs_6h.groupby([pd.Grouper(freq='6H'), 'DOLocationID']).count()

        return dropoffs_6h

    def convert_to_demand_dataset(self):
        df_input_clean = self.clean_data()
        df_input_clean['tpep_pickup_datetime'] = pd.to_datetime(df_input_clean['tpep_pickup_datetime'], format='%Y-%m-%d %H:%M:%S')
        df_input_clean['tpep_dropoff_datetime'] = pd.to_datetime(df_input_clean['tpep_dropoff_datetime'], format='%Y-%m-%d %H:%M:%S')

        pickups_6h_df = self.number_of_pickups_in6hour(df_input_clean).reset_index(level=['PULocationID'])
        dropoffs_6h_df = self.number_of_dropoffs_in6hour(df_input_clean).reset_index(level=['DOLocationID'])

        df_input_clean.set_index(df_input_clean['tpep_pickup_datetime'], inplace=True)

        # Resample input dataset to timestamp for each PULocationID
        df_demand = df_input_clean.groupby([pd.Grouper(freq=f'{self.timestamp}H'), 'PULocationID']).apply(self.demand_calculation, pickups_6h_df, dropoffs_6h_df)
        df_demand = df_demand.to_frame(name='demand').reset_index(level=['PULocationID'])

        df_demand = df_demand.reset_index()
        df_demand = df_demand.rename(columns = {'tpep_pickup_datetime':'timestamp'})

        df_demand['timestamp'] = pd.to_datetime(df_demand['timestamp'], format='%Y-%m-%d %H:%M:%S')

        # only keep known PULocationIDs. 264 and 265 are unknown ids.
        df_demand = df_demand.query("PULocationID <= 263")
        df_demand = self.add_rows_with_zero_demand(df_demand, pickups_6h_df, dropoffs_6h_df)

        # add 1 to every demands to avoid zero demand
        df_demand.loc[:, "demand"] += 1
        df_demand = df_demand.reset_index(drop=True)
        df_demand = df_demand.sort_values(["timestamp", "PULocationID"])

        return df_demand


    # handle outliers and missing values
    def clean_data(self):
        df_input_clean = data_preprocessing.Preprocessing(self.df_input).fit_transform() # TODO: add dir of input df
        return df_input_clean

    def add_rows_with_zero_demand(self, df: pd.DataFrame, pickups_6h_df, dropoffs_6h_df):
        df_grouped = df.groupby("timestamp")
        group_keys = list(df_grouped.groups.keys())
        po_loc_ids = np.arange(1, CONFIG.get("number_of_zones") + 1)
        for i in group_keys:
            df1 = df_grouped.get_group(i)
            current_pu_loc = df1["PULocationID"].to_numpy()
            ids_not_in_po_loc = np.setdiff1d(po_loc_ids , current_pu_loc).tolist()
            demand = []

            for id in ids_not_in_po_loc:

                pickups_6h_def_filtered = pickups_6h_df.loc[(pickups_6h_df.index  < df1['timestamp'].iloc[0]) & (pickups_6h_df.PULocationID == id)]
                dropoffs_6h_df_filtered = dropoffs_6h_df.loc[(dropoffs_6h_df.index  < df1['timestamp'].iloc[0]) & (dropoffs_6h_df.DOLocationID == id)]

                if len(pickups_6h_def_filtered) > 0 and len(dropoffs_6h_df_filtered) > 0:
                    diff = dropoffs_6h_df_filtered.iloc[-1]['tpep_pickup_datetime'] - pickups_6h_def_filtered.iloc[-1]['tpep_dropoff_datetime']

                    if diff > 0 :
                        demand.append(diff + 1)
                    else:
                        demand.append(1)

                else:
                    demand.append(1)

            n = len(ids_not_in_po_loc)
            new_data = {'timestamp': [df1['timestamp'].iloc[0]] * n, 'PULocationID': ids_not_in_po_loc, 'demand': demand}
            new_rows_df = pd.DataFrame(new_data)
            df = pd.concat([df, new_rows_df], ignore_index=True)
        return df



class PredictionDF:
    def __init__(self, needed_datetime, df_demand, timestamp=3, deep=False):
        self.timestamp = timestamp
        self.needed_datetime = needed_datetime
        self.df_demand = df_demand  # demand df calculated from cleaned input df
        self.df_predict = None      # df that contains needed timestamp and 263 pulocationids
        self.df_features = None     # df_predict with new features
        self.deep = deep

    def add_all_features(self):
        self.create_df_for_predict()
        self.add_weather()
        self.label_encoding()
        self.encode_time()
        self.add_weekend()
        self.add_holiday()
        self.add_rolling()
        self.add_lags()
        self.target_encode_puid()
        return self.df_features

    # create 263 rows for one timestamp that is gonna be used in prediction
    def create_df_for_predict(self):
        pulocationid_list = list(range(1, CONFIG.get("number_of_zones") + 1))

        # Create a timestamp column with similar values (e.g., '2023-08-07 12:00:00')
        timestamp_value = pd.to_datetime(self.needed_datetime, format='%Y-%m-%d %H:%M:%S')
        timestamp_list = [timestamp_value] * len(pulocationid_list)

        self.df_predict = pd.DataFrame({'timestamp': timestamp_list, 'PULocationID': pulocationid_list})


    def add_weather(self):
        # find the weather for the next timestamp
        self.df_features = weather.Weather(self.df_predict).get_weather_csv(self.needed_datetime)

    def label_encoding(self):
        # Create a label encoder object
        le = LabelEncoder()

        # Fit the label encoder to the categorical column
        self.df_features['Borough'] = le.fit_transform(self.df_features['Borough'])
        self.df_features['icon'] = le.fit_transform(self.df_features['icon'])

    # use cyclic time as feature
    def encode_time(self):
        self.df_features.reset_index(inplace=True)
        self.df_features["seconds"] = self.df_features["timestamp"].map(pd.Timestamp.timestamp)

        def sin_transformer(period):
            return FunctionTransformer(lambda x: np.sin(x / period * 2 * np.pi))

        def cos_transformer(period):
            return FunctionTransformer(lambda x: np.cos(x / period * 2 * np.pi))

        seconds_in_day = 24 * 60 * 60
        seconds_in_week = 7 * seconds_in_day

        self.df_features["time_sin"] = sin_transformer(seconds_in_day).fit_transform(self.df_features["seconds"])
        self.df_features["time_cos"] = cos_transformer(seconds_in_day).fit_transform(self.df_features["seconds"])

        self.df_features["week_sin"] = sin_transformer(seconds_in_week).fit_transform(self.df_features["seconds"])
        self.df_features["week_cos"] = cos_transformer(seconds_in_week).fit_transform(self.df_features["seconds"])

        self.df_features.drop('seconds' , axis=1, inplace=True)
        self.df_features.set_index("timestamp", inplace=True)

    def add_weekend(self):
        self.df_features["week_day"] = self.df_features.index.dayofweek

    def add_holiday(self):
        self.df_features.reset_index(inplace=True)
        cal = calendar()
        holidays = cal.holidays(start=self.df_features.timestamp.min(), end=self.df_features.timestamp.max())
        self.df_features['holiday'] = self.df_features['timestamp'].isin(holidays).astype(int)
        self.df_features.loc[self.df_features.timestamp.dt.dayofweek >= 5, 'holiday'] = 1
        self.df_features = self.df_features.set_index("timestamp")

    """
    GET DEMAND FOR ROLLING FEATURE FROM DF_DEMAND. BECAUSE WE NEED REAL DEMAND.
    """
    def add_rolling(self, window=2):
        self.df_demand = self.df_demand.set_index("timestamp")

        df_all = pd.concat([self.df_demand, self.df_features], axis=0)
        df_all = df_all.fillna(0)

        grouped_df = df_all.groupby('PULocationID')

        # calculate the rolling mean and standard deviation using a window for each group
        rolling_mean = grouped_df["demand"].rolling(window=window,closed="left").mean().reset_index() 
        rolling_std = grouped_df["demand"].rolling(window=window, closed="left").std().reset_index()

        # rename the columns of the resulting dataframes
        rolling_mean.columns = ['PULocationID', 'timestamp', 'rollingMean']
        rolling_std.columns = ['PULocationID', 'timestamp', 'rollingStd']
        
        rolling_mean["timestamp"] = pd.to_datetime(rolling_mean["timestamp"], format='%Y-%m-%d %H:%M:%S')
        rolling_std["timestamp"] = pd.to_datetime(rolling_std["timestamp"], format='%Y-%m-%d %H:%M:%S')

        df_all = df_all.reset_index()
        

        df_all = pd.merge(df_all, rolling_mean, on=['timestamp', 'PULocationID'], how='left')
        df_all = pd.merge(df_all, rolling_std, on=['timestamp', 'PULocationID'], how='left')

        df_all = self.fillna_rolling(df_all, "rollingMean")
        df_all = self.fillna_rolling(df_all, "rollingStd")
        
        self.df_features = df_all.tail(self.df_features.shape[0]).copy()
        self.df_features = self.df_features.drop("demand", axis=1)
        


    def fillna_rolling(self, df, col):
        grouped = df.groupby("PULocationID")
        df[col] = grouped[col].ffill()
        return df

    def add_lags(self):
        df_all = pd.concat([self.df_demand, self.df_features], axis=0)
        # 1h lagging
        number_of_lags = 8  # lags shall be tuned
        grouped = df_all.groupby("PULocationID")
        for i in range(1, number_of_lags + 1):
            df_all['demand_lagged'+str(i)] = grouped["demand"].shift(i)

        # fill nan lags
        grouped = df_all.groupby("PULocationID")
        for i in range(1, number_of_lags + 1):
            df_all['demand_lagged'+str(i)] = grouped['demand_lagged'+str(i)].ffill()

        # one week lag
        grouped = df_all.groupby("PULocationID")
        df_all['demand_lagged_weekly'] = grouped["demand"].shift(7*24/int(self.timestamp))
        df_all["demand_lagged_weekly"] = grouped["demand_lagged_weekly"].ffill()

        self.df_features = df_all.tail(self.df_features.shape[0]).copy()
        self.df_features = self.df_features.drop("demand", axis=1)


    def target_encode_puid(self):
        if not self.deep:
            # use encoded pulocationid of train data 
            relative_path = CONFIG.get("target_encoded_zones_dir")
            mean_demand_per_location = pd.read_pickle(relative_path)
            self.df_features['PULocationID'] = self.df_features['PULocationID'].map(mean_demand_per_location)
            self.df_features = self.df_features.set_index("timestamp")
        else:
            # use encoded pulocationid of train data 

            location_features_df = pd.read_parquet("api/utils/locations_latent_features.parquet")
            self.df_features = pd.merge(self.df_features, location_features_df, left_on='PULocationID', right_on='PULocationID',)

            relative_path = 'api/utils/mean_demand_per_location.pkl'
            mean_demand_per_location = pd.read_pickle(relative_path)
            self.df_features['PULocationID_encoded'] = self.df_features['PULocationID'].map(mean_demand_per_location)
            self.df_features = self.df_features.set_index("timestamp")

