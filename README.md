# BeerVolumeProphet
Predict Beer Volume 1 month in advance, for 1 SKU and 1 Distributor. Utilizes 3 separate Meta Prophet algorithms to prepare the future prediction, and one Meta Prophet to make the final predict.  Data analyzed for preparation using KMeans, WCSS Elbow, and PCA for feature resuction.

Cross validation for all Prophets can be ran when hastags are removed, in addition to other lines of code.  Meta Prophet adjustments are hyper-parameter tuned to have their MAPE reduced as much as possible.

Regarding the "A1S2 Meta Prophet Tuning", it focuses on predicting Distributor 1's SKU #2 (rather than SKU 1). Adjust Meta Prophet to be more accurate by adjusting parameters.  It also includes all of the Meta Prophet coding and data requirements without the data preparation, allowing for adjusting for cross-validation runs and not having to run the other stuff.

When finding the best hyper-parameter, keep in mind that every option added for each prior scale adds a dimension.  So the code will have to run every single possibility:

i.e.: below, there are 3 values for each prior scale.  3 x 3 x 3 = 27 total runs.  This can take a long time to run, so choose your hyper-parameter tuning wisely.

### Analyze best hyperparameter tuning for the a1s1 Meta Prophet
a1s1_param_grid = {  
    'changepoint_prior_scale': [0.01, 0.1, 1.6],
    'seasonality_prior_scale': [0.01, 0.1, 1.0],
    'changepoint_range': [0.75, 0.8, 0.95],
}
# Generate all combinations of parameters, for a1s1 Prophet
a1s1_all_params = [dict(zip(a1s1_param_grid.keys(), 
                                a1s1)) for a1s1 in itertools.product(*a1s1_param_grid.values())]
a1s1_mapes = []  # Store the RMSEs for each params here
# Use cross validation to evaluate all Agency 1 and SKU 1 parameters 
#Remove/swap hastags below to cross evaluate numerous variables in
#changepoint, seasonality, and range in param grid above.
for a1s1params in a1s1_all_params:
    a1s1_prophet = Prophet(**a1s1params).fit(a1s1_prophet_feed) 
    # Fit model with given params
    a1s1_crossval = cross_validation(a1s1_prophet, period='31 days', horizon = '31 days')
    a1s1_performance = performance_metrics(a1s1_crossval, rolling_window=1)
    a1s1_mapes.append(a1s1_performance['mape'].values[0])
a1s1_tuning_results = pd.DataFrame(a1s1_all_params)
a1s1_tuning_results['mape'] = a1s1_mapes
best_a1s1_params = a1s1_all_params[np.argmin(a1s1_mapes)]
print(best_a1s1_params)
