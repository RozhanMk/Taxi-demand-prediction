import os
import time
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL
from sklearn.preprocessing import PowerTransformer
from statsmodels.tsa.stattools import adfuller

class TimeAnalysis:
  def __init__(self, df: pd.DataFrame):
    self.df = df['demand'].to_frame(name='demand')

  def reset_df(self, df):
    self.df = df

  """Use ADF test to recognize whether the dataset is stionary or non-stationary"""
  def stationary_score(self, data_frame=None):

    """If data_frame argument is None, the self.df will be used and the result will be saved as class attributes"""
    if data_frame is None: df =self.df
    else: df = data_frame

    result = adfuller(df)
    test_dict = {
        'adf': result[0],
        'p_value': result[1],
        'lags': result[2],
        'n_observations': result[3],
        'critical_values': result[4]
    }

    """
    The dataset can be considered stationary if the T score is greater than the absolute value obtained by taking the area under 
    the distribution curve that corresponds to the 5% level of significance.
    """
    stationary: bool = abs(test_dict.get('adf')) > abs(test_dict.get('critical_values')['5%'])

    if data_frame is None:
      self.adf_dict = test_dict

    return stationary, test_dict



  """
  Using this type of normalization can address the issue of having non-stationary variance. If the dataset is already stationary, 
  this normalization method will have little effect on the dataset.
  """
  def boxcox_normalizer(self):
      # apply Box-Cox transformation
      self.normal_df = self.df.copy()
      self.pt = PowerTransformer(method='box-cox')
      x_transformed = self.pt.fit_transform(self.df)
      self.normal_df['demand'] = x_transformed

      return self.normal_df

  def denormaliztion(self, x):
    return self.pt.inverse_transform(x)


  """The decomposition method decomposes the time-series data into trend, seasonal, and residuals."""
  def decomposition(self, period=24*60, normal=False, plot_b = False):
    if normal==True: df = self.boxcox_normalizer()
    else: df = self.df

    stl = STL(df, period=period)
    res = stl.fit()
    self.trend = res.trend
    self.seasonal = res.seasonal
    self.residuals = res.resid

    if plot_b == True: fig = res.plot()

    return {
            'trend':res.trend,
            'seasonal': res.seasonal,
            'residuals': res.resid,
          }


