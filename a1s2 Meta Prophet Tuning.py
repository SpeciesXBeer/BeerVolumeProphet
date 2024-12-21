# -*- coding: utf-8 -*-
"""
Created on Sun Nov 20 13:22:27 2022

@author: beauw
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import itertools
from pandas import to_datetime
from prophet import Prophet
from pandas import DataFrame
from matplotlib import pyplot
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from prophet.diagnostics import cross_validation
from prophet.diagnostics import performance_metrics
from prophet.plot import plot_cross_validation_metric
from sklearn.decomposition import PCA


# Import raw data sets. Then, combine cariables into one mother dataframe.
#########################################################################
# Import demographics of customers living in the region of each wholesaler.
demographics = pd.read_csv()

# Major events - such as holidays, Superbowl, big soccer games, etc.
major_events = pd.read_csv()
# Change YearMonth to Date/Time
major_events['YearMonth'] = pd.to_datetime(
    major_events['YearMonth'], format='%Y%m')

# Historical volume of each SKU
historical_volume = pd.read_csv()
# Change YearMonth to Date/Time
historical_volume['YearMonth'] = pd.to_datetime(
    historical_volume['YearMonth'], format='%Y%m')

# Overall industry soda sales
industry_soda_sales = pd.read_csv()
# Change YearMonth to Date/Time
industry_soda_sales['YearMonth'] = pd.to_datetime(
    industry_soda_sales['YearMonth'], format='%Y%m')

# Overall industry beer volume
industry_volume = pd.read_csv()
# Change Yearmonth to Date/Time
industry_volume['YearMonth'] = pd.to_datetime(
    industry_volume['YearMonth'], format='%Y%m')

# Any promotions matched up to Year Month
price_sales_promotion = pd.read_csv()
# Change YearMonth to Date/Time
price_sales_promotion['YearMonth'] = pd.to_datetime(
    price_sales_promotion['YearMonth'], format='%Y%m')

# Average temperature of YearMonth in relation to each wholesaler's region
weather = pd.read_csv()
# Change YearMonth to Date/Time
weather['YearMonth'] = pd.to_datetime(weather['YearMonth'], format='%Y%m')

# Merge all variables that depend on SKUs into one data frame - stacking
# on top of Agency, SKU, and then YearMonth
sku_dataframe = historical_volume.merge(
    price_sales_promotion, on=['Agency', 'SKU', 'YearMonth'], how='left')
sku_dataframe = sku_dataframe.merge(
    industry_soda_sales, on=['YearMonth'], how='left')
sku_dataframe = sku_dataframe.merge(
    industry_volume, on=['YearMonth'], how='left')
sku_dataframe = sku_dataframe.merge(major_events, on=['YearMonth'], how='left')

# Merge all variables that depend on Agencies (AKA distributors) by eliminating duplicates
Agency_dataframe = weather.merge(demographics, on=['Agency'], how='left')
# Let's take a look at all the Agencies
#week4_dataframe_agencies = Agency_dataframe.copy()
#week4_dataframe_agencies = week4_dataframe_agencies.groupby('Agency')
# This does not perform well in the Spyder IDE

# Merge both major dataframes (ones depending on SKUs and on Agencies) into one big dataframe
mother_dataframe = sku_dataframe.merge(
    Agency_dataframe, on=['YearMonth', 'Agency'], how='left')

# Turn the categorical SKU data into booleans columns instead. Also making
#a data frame for a PCA run.
PCAmother_df = mother_dataframe.copy()
mother_dataframe = pd.get_dummies(
    mother_dataframe, columns=['SKU'], dummy_na=False)

# Check on null values in the newly formed large dataframe. Let's also check
# out the statistics.
mother_dataframe.isnull().sum()

# Import the testing data now...
testing_dataframe = pd.read_csv()

# Visualize variables graphically that may relate with volume
# plt.scatter(mother_dataframe['Avg_Max_Temp'],mother_dataframe['Volume'])
# plt.scatter(mother_dataframe['Promotions'],mother_dataframe['Volume'])

# Let's drop the Fifa World Cup and Football Gold cup due to 0 value
# contributions.
mother_dataframe.drop(
    columns=['FIFA U-17 World Cup', 'Football Gold Cup'], inplace=True)

#Making a data frame for just SKU1 and Agency 1
#agency1_SKU1_df = mother_dataframe.copy()
#agency1_SKU1_df.query(
 #   'Agency == "Agency_01" and SKU_SKU_01 == 1', inplace=True)
#agency1_SKU1_df.drop('SKU_SKU_02', axis=1, inplace=True)
#agency1_SKU1_df.drop('SKU_SKU_01', axis=1, inplace=True)

#Making a data frame for just SKU2 and Agency 1
agency1_SKU2_df = mother_dataframe.copy()
agency1_SKU2_df.query(
    'Agency == "Agency_01" and SKU_SKU_02 == 1', inplace=True)
agency1_SKU2_df.drop('SKU_SKU_02', axis=1, inplace=True)
agency1_SKU2_df.drop('SKU_SKU_01', axis=1, inplace=True)
#####################################################################
#####################################################################
#####################################################################
######################################################################
# Create a factor plot against time and volume with various variables
#sns.catplot(x ='YearMonth', y ='Volume', data = mother_dataframe)
#sns.catplot(x ='Price', y ='Volume', data = mother_dataframe)
#sns.catplot(x ='Promotions', y ='Volume', data = mother_dataframe)
#sns.catplot(x ='Avg_Population_2017', y ='Volume', data = mother_dataframe)
#sns.catplot(x ='Avg_Yearly_Household_Income_2017', y ='Volume', data = mother_dataframe)
# These all took a very long time to process. Saved plot pictures for later use.
######################################################################
######################################################################
######################################################################
######################################################################
######################################################################
# Creating 3 separate Prophet algorithms, which will make a new dataframe
# with industry volume, soda volume, and avg temperature.
### in order to prepare Prophet for making a prediction of SKU 1 and Agency 1
prophet_feed_df = mother_dataframe.copy()
prophet_feed_soda = prophet_feed_df[['YearMonth', 'Soda_Volume']]
prophet_feed_industry = prophet_feed_df[['YearMonth', 'Industry_Volume']]

# For the weather forecast, we will need to train algorithms on all of
# agency 1's data only (regardless of SKU. Filtering out the rest of the agencies...
prophet_feed_weather = prophet_feed_df[['YearMonth', 'Avg_Max_Temp', 'Agency']]
prophet_feed_weather.query('Agency == "Agency_01"', inplace=True)
prophet_feed_weather.drop('Agency', axis=1, inplace=True)

# Assign Prophet friendly names to variables in both data sets.
# Change time to date-time format.
prophet_feed_soda.columns = ['ds', 'y']
prophet_feed_soda['ds'] = to_datetime(prophet_feed_soda['ds'])
prophet_feed_industry.columns = ['ds', 'y']
prophet_feed_industry['ds'] = to_datetime(prophet_feed_industry['ds'])
prophet_feed_weather.columns = ['ds', 'y']
prophet_feed_weather['ds'] = to_datetime(prophet_feed_weather['ds'])

# Label the Meta Prophet algorithm for each variable
industry_prophet = Prophet()
industry_prophet.fit(prophet_feed_industry)
soda_prophet = Prophet()
soda_prophet.fit(prophet_feed_soda)
weather_prophet = Prophet()
weather_prophet.fit(prophet_feed_weather)



# Combine all futures data and evaluate the three Prophets' predictions.
#### Build a Future forecast dataframe for the soda prophet predict.
sodafuture = list()
for s in range(1, 13):
    sodadate = '2018-%02d' % s
    sodafuture.append([sodadate])
sodafuture = DataFrame(sodafuture)
sodafuture.columns = ['ds']
sodafuture['ds'] = to_datetime(sodafuture['ds'])
#Build Soda Meta Prophet model
### Insert top rated parameters for Soda model
soda_param_grid = {  
  'changepoint_prior_scale': [0.0001],#This is the lowest value in MAPE reduction
  'seasonality_prior_scale': [0.001],#This is the lowest value in MAPE reduction
}
soda_all_params = [dict(zip(soda_param_grid.keys(), 
                                sod)) for sod in itertools.product(*soda_param_grid.values())]
for sparams in soda_all_params:
    soda_prophet = Prophet(**sparams).fit(prophet_feed_soda) 
# Make Soda prediction dataframe.
sodaforecast = soda_prophet.predict(sodafuture)
# Plot the overall beer soda prediction from Soda Prophet
soda_prophet.plot(sodaforecast)
pyplot.show()
# Evaluate performance of the Soda Prophet
soda_crossval = cross_validation(soda_prophet, initial='1095 days', period='31 days', horizon = '365 days')
soda_prophet_performance = performance_metrics(soda_crossval)
soda_fig_performance = plot_cross_validation_metric(soda_crossval, metric='mape')



#### Build a Future forecast dataframe for the industry prophet predict.
industryfuture = list()
for b in range(1, 13):
    industrydate = '2018-%02d' % b
    industryfuture.append([industrydate])
industryfuture = DataFrame(industryfuture)
industryfuture.columns = ['ds']
industryfuture['ds'] = to_datetime(industryfuture['ds'])
#Build Industry Meta Prophet model
### Insert top rated parameters for Industry model
industry_param_grid = {  
    'changepoint_prior_scale': [0.0001], #This is the lowest value in MAPE reduction
    'seasonality_prior_scale': [0.001], #This is the lowest value in MAPE reduction
}
industry_all_params = [dict(zip(industry_param_grid.keys(), 
                                ind)) for ind in itertools.product(*industry_param_grid.values())]
for iparams in industry_all_params:
    industry_prophet = Prophet(**iparams).fit(prophet_feed_industry) 
# Make industry prediction dataframe.
industryforecast = industry_prophet.predict(industryfuture)
# Plot the overall beer industry prediction from iIndustry Prophet
industry_prophet.plot(industryforecast)
pyplot.show()
# Evaluate performance of the industry Prophet
industry_crossval = cross_validation(industry_prophet, initial='1095 days', period='31 days', horizon = '365 days')
industry_prophet_performance = performance_metrics(industry_crossval)
industry_fig_performance = plot_cross_validation_metric(industry_crossval, metric='mape')



# Build a Future forecast dataframe for the weather prophet predict.
weatherfuture = list()
for c in range(1, 13):
    weatherdate = '2018-%02d' % c
    weatherfuture.append([weatherdate])
weatherfuture = DataFrame(weatherfuture)
weatherfuture.columns = ['ds']
weatherfuture['ds'] = to_datetime(weatherfuture['ds'])
#Build weather Meta Prophet model
### Insert top rated parameters for weather model
weather_param_grid = {  
    'changepoint_prior_scale': [0.01],#This is the lowest value in MAPE reduction
    'seasonality_prior_scale': [0.01],#This is the lowest value in MAPE reduction
    'holidays_prior_scale': [0.0001],
}
weather_all_params = [dict(zip(weather_param_grid.keys(), 
                                wet)) for wet in itertools.product(*weather_param_grid.values())]
for wparams in weather_all_params:
    weather_prophet = Prophet(**wparams).fit(prophet_feed_weather) 
# Make weather prediction dataframe.
weatherforecast = weather_prophet.predict(weatherfuture)
# Plot the overall beer weather prediction from weather Prophet
weather_prophet.plot(weatherforecast)
pyplot.show()
#Crossval weather Prophet
weatherforecast = weather_prophet.predict(weatherfuture)
weather_crossval = cross_validation(weather_prophet,initial='1095 days', period='31 days', horizon = '365 days')
weather_prophet_performance = performance_metrics(weather_crossval)
weather_fig_performance = plot_cross_validation_metric(weather_crossval, metric='mape')
#########################################################################

# Start merging all predictions onto one data frame, 
#and change names of columns for final volume predict.
Futures2_df = weatherforecast[['ds', 'yhat']]
Futures2_df = Futures2_df.rename(columns={'yhat': 'Avg_Max_Temp'})
Futures2_df.insert(2, 'yhat', industryforecast['yhat'])
Futures2_df = Futures2_df.rename(columns={'yhat': 'Industry_Volume'})
Futures2_df.insert(3, 'yhat', sodaforecast['yhat'])
Futures2_df = Futures2_df.rename(columns={'yhat': 'Soda_Volume'})
Futures2_df = Futures2_df.rename(columns={'YearMonth': 'ds'})

##########################################################################
##Here is the most important part of the whole coding: the last prophet
#That will predict volume based on other prophet algorithm results.
a1s2_prophet_feed = agency1_SKU2_df[['YearMonth','Volume','Avg_Max_Temp',
                                       'Industry_Volume',
                                       'Soda_Volume']]

a1s2_prophet_feed = a1s2_prophet_feed.rename(columns={'YearMonth': 'ds'})
a1s2_prophet_feed = a1s2_prophet_feed.rename(columns={'Volume': 'y'})
a1s2_prophet = Prophet()
a1s2_prophet.add_regressor('Avg_Max_Temp')
a1s2_prophet.add_regressor('Industry_Volume')
a1s2_prophet.add_regressor('Soda_Volume')


### Analyze best hyperparameter tuning for the a1s2 Meta Prophet
a1s2_param_grid = {  
    'changepoint_prior_scale': [1.6],
    'seasonality_prior_scale': [0.1],
    #'changepoints': ['2013-10-01','2014-10-01','2015-10-01','2016-10-01','2017-10-01'],
    #'seasonality_mode': ['multiplicative'],
    'changepoint_range': [0.95],
}
# Generate all combinations of parameters, for a1s2 Prophet
a1s2_all_params = [dict(zip(a1s2_param_grid.keys(), 
                                a1s2)) for a1s2 in itertools.product(*a1s2_param_grid.values())]
#a1s2_mapes = []  # Store the RMSEs for each params here
# Use cross validation to evaluate all Agency 1 and SKU 1 parameters
for a1s2params in a1s2_all_params:
    a1s2_prophet = Prophet(**a1s2params).fit(a1s2_prophet_feed) 
    
a1s2forecast = a1s2_prophet.predict(Futures2_df)
#Plot the overall volume prediction from a1s2 Prophet
a1s2_prophet.plot(a1s2forecast)
pyplot.show()
#Crossval a1s2 Prophet
a1s2forecast = a1s2_prophet.predict(Futures2_df)
a1s2_crossval = cross_validation(a1s2_prophet, initial='1095 days', period='31 days', horizon = '31 days')
a1s2_prophet_performance = performance_metrics(a1s2_crossval)
a1s2_fig_performance = plot_cross_validation_metric(a1s2_crossval, metric='mape')



