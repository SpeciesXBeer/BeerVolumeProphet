# BeerVolumeProphet
Predict Beer Volume 1 month in advance, for 1 SKU and 1 Distributor. Utilizes 3 separate Meta Prophet algorithms to prepare the future prediction, and one Meta Prophet to make the final predict.  Data analyzed for preparation using KMeans, WCSS Elbow, and PCA for feature resuction.

Cross validation for all Prophets can be ran via the "Weather Prophet" example. Meta Prophet adjustments are hyper-parameter tuned to have their MAPE reduced as much as possible.

Regarding the "A1S2 Meta Prophet Tuning", it focuses on predicting Distributor 1's SKU #2 (rather than SKU 1). Adjust Meta Prophet to be more accurate by adjusting parameters.  It also includes all of the Meta Prophet coding and data requirements without the data preparation, allowing for adjusting for cross-validation runs and not having to run the other stuff.

When finding the best hyper-parameter, keep in mind that every option added for each prior scale adds a dimension.  So the code will have to run every single possibility.

