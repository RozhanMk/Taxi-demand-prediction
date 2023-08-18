# Taxi demand prediction
this repository contains the main project of Rahnema College's machine learning boot camp. The purpose of this project is to predict the demand of 263 zones of NYC 
in the future (for any time interval). As part of this plan, the team seeks to identify areas with higher travel demand to increase the number of available 
drivers in those locations. Incentivizing drivers is a marketing initiative, and predicting demand during different time frames is crucial for implementing this 
process. 

**To understand the general idea check our [presentation](https://github.com/RozhanMk/Taxi-demand-prediction/blob/main/Presentation/Demand%20Prediction.pdf)**

# Dataset
[TLC NYC dataset](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page) contains trip data about 12,672,737 trips which are made between 263 different zones in NYC. The trip records include fields capturing pick-up and 
drop-off dates/times, pick-up and drop-off locations, trip distances, itemized fares, rate types, payment types, and driver-reported passenger counts. The pick-up 
and drop-off  contains the date and time for each trip respectively which are about trips made between 2023/1/1 to 2023/4/30 and the times follow the HH:MM:SS format.

# Demand dataset
We created a demand dataset based on this definition for each timestamp and LocationID:
![](https://github.com/RozhanMk/Taxi-demand-prediction/blob/main/images/demand.png)

# Feature Selection
We added these features to the demand dataset:
| Features | Extra info |
| --- | --- |
| [Weather](https://www.visualcrossing.com/weather/weather-data-services) | temperature and icon from weather API |
| [Cyclical time](https://ianlondon.github.io/blog/encoding-cyclical-features-24hour-time/) | daily and weekly |
| Rolling window feature | rolling std and rolling mean |
| Demand lags | daily and weekly lags |
| PULocationID | target encoding | 
| Borough | |
| Week day | |

# Base Model
We transformed a time series problem into a supervised learning problem. So we can use XGBoost.
![](https://github.com/RozhanMk/Taxi-demand-prediction/blob/main/images/xgboost.png)

# Final Model
We used simple deep learning with dense layers and we got the best result:
![](https://github.com/RozhanMk/Taxi-demand-prediction/blob/main/images/deep.png)

# How to run
Before you run this API, you must get your API key from https://www.visualcrossing.com/weather/weather-data-services and put it in [weather API file](https://github.com/RozhanMk/Taxi-demand-prediction/blob/main/Deploy/api/MLpipline/weather/weather.py).
1. clone the repo:
 ```
   git clone https://github.com/RozhanMk/Taxi-demand-prediction
   cd Taxi-demand-prediction
```
2. Run Django:
```
   cd Deploy
   pip install -r requirements.txt
   python manage.py runserver 
```
   
