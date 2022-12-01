# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 06:20:43 2022

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


# Import raw data sets. Then, combine variables into one mother dataframe.
#########################################################################
# Import demographics of customers living in the region of each wholesaler.
demographics = pd.read_csv(
    r'C:\Users\beauw\OneDrive\Desktop\Machine Learning\OSU - Data Mining Project\demographics.csv')

# Major events - such as holidays, Superbowl, big soccer games, etc.
major_events = pd.read_csv(
    r'C:\Users\beauw\OneDrive\Desktop\Machine Learning\OSU - Data Mining Project\event_calendar.csv')
# Change YearMonth to Date/Time
major_events['YearMonth'] = pd.to_datetime(
    major_events['YearMonth'], format='%Y%m')

# Historical volume of each SKU
historical_volume = pd.read_csv(
    r'C:\Users\beauw\OneDrive\Desktop\Machine Learning\OSU - Data Mining Project\historical_volume.csv')
# Change YearMonth to Date/Time
historical_volume['YearMonth'] = pd.to_datetime(
    historical_volume['YearMonth'], format='%Y%m')

# Overall industry soda sales
industry_soda_sales = pd.read_csv(
    r'C:\Users\beauw\OneDrive\Desktop\Machine Learning\OSU - Data Mining Project\industry_soda_sales.csv')
# Change YearMonth to Date/Time
industry_soda_sales['YearMonth'] = pd.to_datetime(
    industry_soda_sales['YearMonth'], format='%Y%m')

# Overall industry beer volume
industry_volume = pd.read_csv(
    r'C:\Users\beauw\OneDrive\Desktop\Machine Learning\OSU - Data Mining Project\industry_volume.csv')
# Change Yearmonth to Date/Time
industry_volume['YearMonth'] = pd.to_datetime(
    industry_volume['YearMonth'], format='%Y%m')

# Any promotions matched up to Year Month
price_sales_promotion = pd.read_csv(
    r'C:\Users\beauw\OneDrive\Desktop\Machine Learning\OSU - Data Mining Project\price_sales_promotion.csv\price_sales_promotion.csv')
# Change YearMonth to Date/Time
price_sales_promotion['YearMonth'] = pd.to_datetime(
    price_sales_promotion['YearMonth'], format='%Y%m')

# Average temperature of YearMonth in relation to each wholesaler's region
weather = pd.read_csv(
    r'C:\Users\beauw\OneDrive\Desktop\Machine Learning\OSU - Data Mining Project\weather.csv')
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
testing_dataframe = pd.read_csv(
    r'C:\Users\beauw\OneDrive\Desktop\Machine Learning\OSU - Data Mining Project\volume_forecast.csv')

# Visualize variables graphically that may relate with volume
# plt.scatter(mother_dataframe['Avg_Max_Temp'],mother_dataframe['Volume'])
# plt.scatter(mother_dataframe['Promotions'],mother_dataframe['Volume'])

# Let's drop the Fifa World Cup and Football Gold cup due to 0 value
# contributions.
mother_dataframe.drop(
    columns=['FIFA U-17 World Cup', 'Football Gold Cup'], inplace=True)

#Making a data frame for just SKU1 and Agency 1
agency1_SKU1_df = mother_dataframe.copy()
agency1_SKU1_df.query(
    'Agency == "Agency_01" and SKU_SKU_01 == 1', inplace=True)
agency1_SKU1_df.drop('SKU_SKU_02', axis=1, inplace=True)
agency1_SKU1_df.drop('SKU_SKU_01', axis=1, inplace=True)
#####################################################################
#####################################################################
#####################################################################
# Create a heatmap of all variables - take a close note of volume correlation
corr = mother_dataframe[mother_dataframe.columns[:21]].corr()
plt.figure(figsize=(12, 12))
sns.heatmap(corr, vmin=-1, cmap='BuPu', annot=True, fmt=".2f")
plt.show()
######################################################################
# Create a factor plot against time and volume with various variables
##THIS TAKES SERIOUS TIME AND CPU USEAGE (Thus the #s)!!
#sns.catplot(x ='YearMonth', y ='Volume', data = mother_dataframe)
#sns.catplot(x ='Price', y ='Volume', data = mother_dataframe)
#sns.catplot(x ='Promotions', y ='Volume', data = mother_dataframe)
#sns.catplot(x ='Avg_Population_2017', y ='Volume', data = mother_dataframe)
#sns.catplot(x ='Avg_Yearly_Household_Income_2017', y ='Volume', data = mother_dataframe)
# These all took a very long time to process. Saved plot pictures for later use.
######################################################################
#Find optimal number of components for ALL data using PCA. I also stacked and
#scaled the SKU data back into one column for this input.
label_encoder = preprocessing.LabelEncoder()
PCAprescaled = PCAmother_df.copy()
PCAprescaled.drop(PCAprescaled.loc[:,'Easter Day':'Music Fest'], axis=1, inplace=True)

SS = StandardScaler()
PCAprescaled['Agency'] = label_encoder.fit_transform(PCAprescaled['Agency'])
PCAprescaled['YearMonth'] = label_encoder.fit_transform(PCAprescaled['YearMonth'])
PCAprescaled['SKU'] = label_encoder.fit_transform(PCAprescaled['SKU'])
PCAscaled = SS.fit_transform(PCAprescaled)
PCAmodel = PCA(random_state=5000).fit(PCAscaled)
plt.plot(PCAmodel.explained_variance_ratio_,
         linewidth = 4)
plt.xlabel('Components')
plt.ylabel('Explained Variance')
plt.show()
#cumulitive run
plt.plot(np.cumsum(PCAmodel.explained_variance_ratio_),
         linewidth = 4)
plt.xlabel('Components')
plt.ylabel('Explained Variance Cumulative')
plt.show() 

#optimal number of components for just SKU1 and Agency 1
PCAprescaled2 = agency1_SKU1_df.copy()
PCAprescaled2.drop(PCAprescaled2.iloc[:,8:17], axis=1, inplace=True)
PCAprescaled2.drop('Agency', axis=1, inplace=True)
SS = StandardScaler()
PCAprescaled2['YearMonth'] = label_encoder.fit_transform(PCAprescaled2['YearMonth'])
PCAscaled2 = SS.fit_transform(PCAprescaled2)
PCAmodel2 = PCA(random_state=5000).fit(PCAscaled2)
plt.plot(PCAmodel2.explained_variance_ratio_,
         linewidth = 4)
plt.xlabel('Components')
plt.ylabel('Explained Variance')
plt.show()
#cumulitive run
plt.plot(np.cumsum(PCAmodel2.explained_variance_ratio_),
         linewidth = 4)
plt.xlabel('Components')
plt.ylabel('Explained Variance Cumulative')
plt.show()
######################################################################
# WCSS Elbow method - then plot KMeans
# After looking at WCSS, the only viable options seem to be pricing
# And promotions.
# Pricing first. I am encoding YearMonth column to include dates as variables
mother_df_Seq = mother_dataframe.copy()
mother_df_Seq0 = mother_dataframe.copy()
label_encoder = preprocessing.LabelEncoder()
mother_df_Seq0['YearMonth'] = label_encoder.fit_transform(mother_df_Seq0['YearMonth'])

price_trans_x = mother_df_Seq0.iloc[:, [1, 2, 3]].values
Standard_Scale = StandardScaler()
Standard_Scale.fit_transform(price_trans_x[:,1:3])
wcss = []
for i in range(1, 11):
    pricekmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    pricekmeans.fit(price_trans_x)
    wcss.append(pricekmeans.inertia_)
plt.figure(figsize=(10, 5))
sns.lineplot(wcss, marker='o', color='red')
plt.title('Elbow Fit')
plt.xlabel('Price - Number of Clusters')
plt.ylabel('WCSS')
plt.show()

# Unique labels for the cluster centroids
price_y_kmeans = KMeans(n_clusters=2, init='k-means++',
                        max_iter=300, n_init=10, random_state=0)
price_z_kmeans = price_y_kmeans.fit_predict(price_trans_x)
price_u_labels = np.unique(price_z_kmeans)
print(price_u_labels)

# Plot the centroids
plt.scatter(price_trans_x[price_z_kmeans == 0, 0],
            price_trans_x[price_z_kmeans == 0, 1], s=100, c='red', label='Cluster 1')
plt.scatter(price_trans_x[price_z_kmeans == 1, 0],
            price_trans_x[price_z_kmeans == 1, 1], s=100, c='blue', label='Cluster 2')
#plt.scatter(price_trans_x[price_z_kmeans == 2, 0],
            #price_trans_x[price_z_kmeans == 2, 1], s=100, c='green', label='Cluster 3')
#plt.scatter(price_trans_x[price_z_kmeans==3, 0], price_trans_x[price_z_kmeans==3, 1], s=100, c='cyan', label ='Cluster 4')
plt.scatter(price_y_kmeans.cluster_centers_[:, 0], price_y_kmeans.cluster_centers_[
            :, 1], s=300, c='yellow', label='Centroids')
plt.title('Clusters of Pricing')
plt.xlabel('Pricing ')
plt.ylabel('Volume')
plt.show()

# Now Promotions..
promo_trans_x = mother_df_Seq0.iloc[:, [1, 2, 5]].values
Standard_Scale.fit_transform(promo_trans_x[[1]])
Standard_Scale.fit_transform(promo_trans_x[[2]])
Standard_Scale.fit_transform(promo_trans_x[[5]])
wcss = []
for i in range(1, 11):
    promokmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    promokmeans.fit(promo_trans_x)
    wcss.append(promokmeans.inertia_)
plt.figure(figsize=(10, 5))
sns.lineplot(wcss, marker='o', color='red')
plt.title('Elbow Fit')
plt.xlabel('Promotions - Number of Clusters')
plt.ylabel('WCSS')
plt.show()

# Unique labels for the cluster centroids
promo_y_kmeans = KMeans(n_clusters=2, init='k-means++',
                        max_iter=300, n_init=10, random_state=0)
promo_z_kmeans = promo_y_kmeans.fit_predict(promo_trans_x)
promo_u_labels = np.unique(promo_z_kmeans)
print(promo_u_labels)

# Plot the centroids
plt.scatter(promo_trans_x[promo_z_kmeans == 0, 0],
            promo_trans_x[promo_z_kmeans == 0, 1], s=100, c='red', label='Cluster 1')
plt.scatter(promo_trans_x[promo_z_kmeans == 1, 0],
            promo_trans_x[promo_z_kmeans == 1, 1], s=100, c='blue', label='Cluster 2')
#plt.scatter(promo_trans_x[promo_z_kmeans == 2, 0],
            #promo_trans_x[promo_z_kmeans == 2, 1], s=100, c='green', label='Cluster 3')
plt.scatter(promo_y_kmeans.cluster_centers_[:, 0], promo_y_kmeans.cluster_centers_[
            :, 1], s=300, c='yellow', label='Centroids')
plt.title('Clusters of Promotions')
plt.xlabel('Promotions')
plt.ylabel('Volume')
plt.show()

# Let's do Sales, Pricing, Promotions, Volume, Yearly Household Income, and
# Average Population via multi-Kmeans clustering. See if all these together
#does anything...

mother_df_Seq = mother_dataframe.copy()
mother_df_Seq.drop(
    mother_df_Seq.loc[:, 'Soda_Volume':'Avg_Max_Temp'], axis=1, inplace=True)
mother_df_Seq.drop(
    mother_df_Seq.loc[:, 'SKU_SKU_01':'SKU_SKU_34'], axis=1, inplace=True)
mother_df_Seq.drop('Agency', axis=1, inplace=True)
mother_df_Seq['YearMonth'] = label_encoder.fit_transform(mother_df_Seq['YearMonth'])
#mother_df_Seq.drop('YearMonth', axis=1, inplace=True)

SS = StandardScaler()
Blob_df = SS.fit_transform(mother_df_Seq.iloc[:,0:7])
blob_trans_x = Blob_df
wcss = []
for i in range(1, 11):
    blobkmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    blobkmeans.fit(blob_trans_x)
    wcss.append(blobkmeans.inertia_)
plt.figure(figsize=(10, 5))
sns.lineplot(wcss, marker='o', color='red')
plt.title('Elbow Fit - Lotta Variables')
plt.xlabel('Lotta Variables - Number of Clusters')
plt.ylabel('WCSS')
plt.show()

cluster_results = pd.DataFrame(Blob_df, columns=['YearMonth','Volume', 'Price', 'Sales',
                              'Promotions', 'Avg_Population_2017', 
                              'Avg_Yearly_Household_Income_2017'])
blob_kmeans = KMeans(n_clusters=4)
y = blob_kmeans.fit_predict(cluster_results[['YearMonth','Volume', 'Price', 'Sales',
                              'Promotions', 'Avg_Population_2017', 
                              'Avg_Yearly_Household_Income_2017']]) 

y2 = pd.DataFrame(y, columns=[0])
cluster_results['Cluster_Results'] = y2

plt.scatter(blob_trans_x[y == 0, 0],
           blob_trans_x[y == 0, 1], s=100, c='red', label='Cluster 1')
plt.scatter(blob_trans_x[y == 1, 0],
            blob_trans_x[y == 1, 1], s=100, c='blue', label='Cluster 2')
plt.scatter(blob_trans_x[y == 2, 0],
            blob_trans_x[y == 2, 1], s=100, c='green', label='Cluster 3')
plt.scatter(blob_trans_x[y == 3, 0],
            blob_trans_x[y == 3, 1], s=100, c='orange', label='Cluster 4')
plt.scatter(blob_kmeans.cluster_centers_[:, 0], blob_kmeans.cluster_centers_[
            :, 1], s=100, c='yellow', label='Centroids')
plt.title('Clusters of a Bunch of Variables')
plt.xlabel('Variables')
plt.ylabel('Y')
plt.show()


# KMeans now completed for Promotions and Pricing.

######################################################################
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
# Plot the overall beer industry prediction from Soda Prophet
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
Futures_df = weatherforecast[['ds', 'yhat']]
Futures_df = Futures_df.rename(columns={'yhat': 'Avg_Max_Temp'})
Futures_df.insert(2, 'yhat', industryforecast['yhat'])
Futures_df = Futures_df.rename(columns={'yhat': 'Industry_Volume'})
Futures_df.insert(3, 'yhat', sodaforecast['yhat'])
Futures_df = Futures_df.rename(columns={'yhat': 'Soda_Volume'})
Futures_df = Futures_df.rename(columns={'YearMonth': 'ds'})

##########################################################################
##Here is the most important part of the whole coding: the last prophet
#That will predict volume based on other prophet algorithm results.
a1s1_prophet_feed = agency1_SKU1_df[['YearMonth','Volume','Avg_Max_Temp',
                                       'Industry_Volume',
                                       'Soda_Volume']]

a1s1_prophet_feed = a1s1_prophet_feed.rename(columns={'YearMonth': 'ds'})
a1s1_prophet_feed = a1s1_prophet_feed.rename(columns={'Volume': 'y'})
a1s1_prophet = Prophet()
a1s1_prophet.add_regressor('Avg_Max_Temp')
a1s1_prophet.add_regressor('Industry_Volume')
a1s1_prophet.add_regressor('Soda_Volume')


### Analyze best hyperparameter tuning for the a1s1 Meta Prophet
a1s1_param_grid = {  
    'changepoint_prior_scale': [1.6],
    'seasonality_prior_scale': [0.1],
    #'changepoints': ['2013-10-01','2014-10-01','2015-10-01','2016-10-01','2017-10-01'],
    #'seasonality_mode': ['multiplicative'],
    'changepoint_range': [0.95],
}
# Generate all combinations of parameters, for a1s1 Prophet
a1s1_all_params = [dict(zip(a1s1_param_grid.keys(), 
                                a1s1)) for a1s1 in itertools.product(*a1s1_param_grid.values())]
#a1s1_mapes = []  # Store the RMSEs for each params here
# Use cross validation to evaluate all Agency 1 and SKU 1 parameters 
#Remove/swap hastags below to cross evaluate numerous variables in
#changepoint, seasonality, and range in param grid above.
for a1s1params in a1s1_all_params:
    a1s1_prophet = Prophet(**a1s1params).fit(a1s1_prophet_feed) 
    # Fit model with given params
    #a1s1_crossval = cross_validation(a1s1_prophet, period='31 days', horizon = '31 days')
    #a1s1_performance = performance_metrics(a1s1_crossval, rolling_window=1)
    #a1s1_mapes.append(a1s1_performance['mape'].values[0])
#a1s1_tuning_results = pd.DataFrame(a1s1_all_params)
#a1s1_tuning_results['mape'] = a1s1_mapes
#best_a1s1_params = a1s1_all_params[np.argmin(a1s1_mapes)]
#print(best_a1s1_params)

a1s1forecast = a1s1_prophet.predict(Futures_df)
#Plot the overall volume prediction from a1s1 Prophet
a1s1_prophet.plot(a1s1forecast)
pyplot.show()
#Crossval a1s1 Prophet
a1s1forecast = a1s1_prophet.predict(Futures_df)
a1s1_crossval = cross_validation(a1s1_prophet, initial='1095 days', period='31 days', horizon = '31 days')
a1s1_prophet_performance = performance_metrics(a1s1_crossval)
a1s1_fig_performance = plot_cross_validation_metric(a1s1_crossval, metric='mape')

#Final prediction 1 month
print(a1s1forecast.head(1))


