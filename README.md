Ames Housing Dataset: Initial Exploratory Data Analysis
The Question
How do housing features in Ames, Iowa correlate with sale prices, and what initial data transformations are necessary before building a predictive machine learning model?

What I Did
Loaded the Ames Housing dataset (train.csv).

Handled missing categorical data by replacing NaN with "Missing" to capture the absence of features (like garages or pools) as a predictive signal.

Visualized the distributions of key variables (SalePrice, GrLivArea, OverallQual, YearBuilt, TotalBsmtSF) using histograms and count plots.

Analyzed correlations between numeric features using a correlation heatmap and scatter plots for the top predictors.

Identified statistical outliers and skewed distributions using box plots.

Five Key Findings
Right-Skewed Target Variable (SalePrice): The target variable is heavily right-skewed, requiring a logarithmic transformation (np.log1p()) to normalize variance for linear models.

High Multicollinearity: Features like GarageCars and GarageArea (r > 0.88), as well as TotalBsmtSF and 1stFlrSF (r = 0.90), are highly collinear. Redundant features must be dropped or combined to ensure stable model coefficients.

Non-Linear Strong Predictor (OverallQual): OverallQual has the highest correlation with SalePrice (r ≈ 0.87) but exhibits an exponential relationship, suggesting the need for polynomial or categorical encoding.

Extreme Outliers in Living Area (GrLivArea): Properties exceeding 4,000 sq ft with unexpectedly low sale prices (< $200k) represent non-standard transactions and must be removed to prevent model distortion.

Extreme Skewness in Lot Dimensions (LotArea): LotArea contains massive positive outliers. Applying a log transformation or binning strategy is necessary to prevent area-based metrics from skewing the model.
