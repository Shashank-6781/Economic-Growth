# Economic Growth — Project Workflow

## Data inspection
Use `head()`, `info()`, `describe(include='all')`, and `isnull().sum()`.

## Missing values
The assigned material describes group-based imputation by `Region`: median for numerical fields and mode for categorical fields such as `Climate`.

## Visualization
The workflow includes correlation heatmaps, bar plots, scatter plots, filtering, and correlation analysis around GDP.

## Encoding
Categorical features are converted into numeric values with `LabelEncoder`.

## Train/test split
`train_test_split` separates training and test data for model evaluation.

## Regression models
- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor

## Evaluation
- R² score
- RMSE

## Deployment
The supplied project selects Random Forest for deployment and uses Flask to load the model, accept user input, predict GDP, and render the output.
