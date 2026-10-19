# Ames Housing Dataset: Initial Exploratory Data Analysis

## The Question
How do housing features in Ames, Iowa correlate with sale prices, and what initial data transformations are necessary before building a predictive machine learning model?

## What I Did
1. Loaded the Ames Housing dataset (`train.csv`).
2. Handled missing categorical data by replacing `NaN` with "Missing" to capture the absence of features (like garages or pools) as a predictive signal.
3. Visualized the distributions of key variables (`SalePrice`, `GrLivArea`, `OverallQual`, `YearBuilt`, `TotalBsmtSF`) using histograms and count plots.
4. Analyzed correlations between numeric features using a correlation heatmap and scatter plots for the top predictors.
5. Identified statistical outliers and skewed distributions using box plots.

## Five Key Findings
1. **Right-Skewed Target Variable (`SalePrice`):** The target variable is heavily right-skewed, requiring a logarithmic transformation (`np.log1p()`) to normalize variance for linear models.
2. **High Multicollinearity:** Features like `GarageCars` and `GarageArea` (r > 0.88), as well as `TotalBsmtSF` and `1stFlrSF` (r = 0.90), are highly collinear. Redundant features must be dropped or combined to ensure stable model coefficients.
3. **Non-Linear Strong Predictor (`OverallQual`):** `OverallQual` has the highest correlation with `SalePrice` (r ≈ 0.87) but exhibits an exponential relationship, suggesting the need for polynomial or categorical encoding.
4. **Extreme Outliers in Living Area (`GrLivArea`):** Properties exceeding 4,000 sq ft with unexpectedly low sale prices (< $200k) represent non-standard transactions and must be removed to prevent model distortion.
5. **Extreme Skewness in Lot Dimensions (`LotArea`):** `LotArea` contains massive positive outliers. Applying a log transformation or binning strategy is necessary to prevent area-based metrics from skewing the model.
