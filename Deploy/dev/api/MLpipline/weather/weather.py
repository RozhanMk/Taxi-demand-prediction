import requests
import pandas as pd
import re
import os

print(os.getcwd())
class Weather:
    def __init__(self, df_predict):
        self.df_predict = df_predict
    

    def get_weather_csv(self, needed_datetime):
        # Weather API key
        key = "YOUR API KEY FROM https://www.visualcrossing.com/weather/weather-data-services"

        locations = ["manhattan", "brooklyn", "usa queens", "bronx", "staten island", "newark"]
        needed_date = needed_datetime.date()
        needed_date = needed_date.strftime("%Y-%m-%d")
        needed_hour = str(needed_datetime.hour)
        if len(needed_hour) == 1:
            needed_hour = "0" + needed_hour

        # empty the file before opening
        with open("api/MLpipline/weather/weather_data.csv", 'w', newline='') as file:
            file.truncate()

        for index, location in enumerate(locations):
            url = f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{location}/{needed_date}T{needed_hour}:00:00?unitGroup=metric&key={key}&contentType=csv&include=current"
            response = requests.get(url)

            if response.status_code == 200:
                csv_data = response.content.decode("utf-8")  
                with open("api/MLpipline/weather/weather_data.csv", "a", newline="", encoding="utf-8") as file:
                    if index == 0:
                        file.write(csv_data)
                    else:
                        file.write(csv_data.split("\n", 1)[1])  # Skip the first line (headers)
                    
            else:
                print(response.text)
                return
        weather = pd.read_csv("api/MLpipline/weather/weather_data.csv")
        weather.rename(columns={"name":"Borough", "datetime":"timestamp"}, inplace=True)
        weather["timestamp"] = pd.to_datetime(weather["timestamp"])

        weather = self.change_borough_name(weather)
        self.map_zones_to_borough()

        df_with_weather = self.merge_demand_and_weather(weather)
        return df_with_weather


    def change_borough_name(self, weather: pd.DataFrame):
        # matching borough name of weather dataset with names in zones dataset
        weather.loc[weather['Borough'].str.contains(re.compile(r'Manhattan', re.IGNORECASE)), 'Borough'] = "Manhattan"
        weather.loc[weather['Borough'].str.contains(re.compile(r'Queens', re.IGNORECASE)), 'Borough'] = "Queens"
        weather.loc[weather['Borough'].str.contains(re.compile(r'Bronx', re.IGNORECASE)), 'Borough'] = "Bronx"
        weather.loc[weather['Borough'].str.contains(re.compile(r'Staten', re.IGNORECASE)), 'Borough'] = "Staten Island"
        weather.loc[weather['Borough'].str.contains(re.compile(r'Brooklyn', re.IGNORECASE)), 'Borough'] = "Brooklyn"
        weather.loc[weather['Borough'].str.contains(re.compile(r'Newark', re.IGNORECASE)), 'Borough'] = "EWR"

        weather = weather[["Borough", "timestamp", "temp", "icon"]]

        return weather

    
    def map_zones_to_borough(self):
    	# zones.csv contains borough for each LocationID in nyc. get it from TLC nyc website.
        zones_df = pd.read_csv("api/MLpipline/weather/zones.csv")
        pulocationid_to_borough = dict(zip(zones_df.LocationID, zones_df.Borough))
        # create Borough column in dataframe
        self.df_predict['Borough'] = self.df_predict['PULocationID'].map(pulocationid_to_borough)

    def merge_demand_and_weather(self, weather: pd.DataFrame):
        # merge weather dataset with demand dataset
        df_with_weather = pd.merge(self.df_predict, weather, on=['Borough',"timestamp"], how="left")
        df_with_weather = df_with_weather.set_index("timestamp")
        df_with_weather = df_with_weather.sort_index()
        return df_with_weather

