# ==========================================
# 1. IMPORT LIBRARIES
# ==========================================
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score

# ==========================================
# 2. LOAD DATASET
# ==========================================
df = pd.read_csv('train.csv')
print(df.head())
print("\nDataset Shape", df.shape)
print("\nColumn Names:", df.columns.to_list())
print("\nDataset Info:", df.info())
print("\nMissing Values:\n", df.isnull().sum())
print("\nStatistical Summary:\n", df.describe())

# ==========================================
# 3. DATA PREPROCESSING / CLEANING
# ==========================================
missing_values = df.isnull().sum()
print("\nColumn with Missing Values:\n", missing_values[missing_values > 0].sort_values(ascending=False))  
categorical_none_columns = [
    "PoolQC",
    "MiscFeature",
    "Alley",
    "Fence",
    "MasVnrType",
    "FireplaceQu",
    "GarageType",
    "GarageFinish",
    "GarageQual",
    "GarageCond",
    "BsmtExposure",
    "BsmtFinType1",
    "BsmtFinType2",
    "BsmtQual",
    "BsmtCond"
]

for column in categorical_none_columns:
    df[column] = df[column].fillna("None")

df["GarageYrBlt"] = df["GarageYrBlt"].fillna(0)
df["MasVnrArea"] = df["MasVnrArea"].fillna(0)
 
df["LotFrontage"] = df["LotFrontage"].fillna(
    df["LotFrontage"].median()
)

df["Electrical"] = df["Electrical"].fillna(
    df["Electrical"].mode()[0]
)

print("\nMissing Values After Preprocessing:")
print(df.isnull().sum()[df.isnull().sum() > 0])

# ==========================================
# 3.1 ENCODING CATEGORICAL VARIABLES
# ==========================================
categorical_columns = df.select_dtypes(
    include=['str', 'object']
).columns

print("\nCategorical Columns:")
print(categorical_columns.tolist())

# One-hot encode categorical variables
df = pd.get_dummies(
    df,
    columns=categorical_columns,
    drop_first=True
)
print("\nDataset Shape After Encoding:")
print(df.shape)

print("\nRemaining Categorical Columns:")
print(df.select_dtypes(include=['str', 'object']).columns.tolist())
print("\nData Types:",df.dtypes.value_counts())

# ==========================================
# 4. FEATURE & TARGET SEPARATION
# ==========================================
X = df.drop("SalePrice", axis=1)
y = df["SalePrice"]

print("\nData Shape:",X.shape)
print("Target Shape:", y.shape)

# ==========================================
# 5. TRAIN / TEST SPLIT
# ==========================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print("\nTrain Data Shape:", X_train.shape)
print("Test Data Shape:", X_test.shape)
print("Train Target Shape:", y_train.shape)
print("Test Target Shape:", y_test.shape)

# ==========================================
# 6. REGRESSION MODELS
# ==========================================

# ------------------------------------------
# LINEAR REGRESSION
# ------------------------------------------
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
linear_predictions = linear_model.predict(X_test)
print("\nLinear Regression Predictions:", linear_predictions[:10])
print("\nLinear Regression Actual Prices:", y_test.iloc[:10].values)
print("\nLinear Regression Predicted Prices:", linear_predictions[:10])

# ------------------------------------------
# LINEAR REGRESSION - EVALUATION
# ------------------------------------------
linear_mae = mean_absolute_error(y_test, linear_predictions)
linear_rmse = mean_squared_error(
    y_test,
    linear_predictions
) ** 0.5
linear_r2 = r2_score(y_test, linear_predictions)

print("\nLINEAR REGRESSION - EVALUATION")
print("Mean Absolute Error:", linear_mae)
print("Root Mean Squared Error:", linear_rmse)
print("R² Score:", linear_r2)

# ------------------------------------------
# DECISION TREE REGRESSION
# ------------------------------------------
tree_model = DecisionTreeRegressor(
    random_state=42
)

tree_model.fit(X_train, y_train)
tree_predictions = tree_model.predict(X_test)
print("\nDecision Tree Predictions:", tree_predictions[:10])
print("\nDecision Tree Actual Prices:", y_test.iloc[:10].values)

tree_mae = mean_absolute_error(y_test,tree_predictions)
tree_rmse = mean_squared_error(y_test,tree_predictions) ** 0.5
tree_r2 = r2_score(y_test,tree_predictions)

print("\nDECISION TREE - EVALUATION")
print("Mean Absolute Error:", tree_mae)
print("Root Mean Squared Error:", tree_rmse)
print("R² Score:", tree_r2)

# ------------------------------------------
# RANDOM FOREST REGRESSION
# ------------------------------------------
forest_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)
forest_model.fit(X_train, y_train)
forest_predictions = forest_model.predict(X_test)
print("\nRandom Forest Predictions:", forest_predictions[:10])
print("\n Random ForestActual Prices:", y_test.iloc[:10].values)

forest_mae = mean_absolute_error(y_test,forest_predictions)
forest_rmse = mean_squared_error(y_test,forest_predictions) ** 0.5
forest_r2 = r2_score(y_test,forest_predictions)

print("\nRANDOM FOREST - EVALUATION")
print("Mean Absolute Error:", forest_mae)
print("Root Mean Squared Error:", forest_rmse)
print("R² Score:", forest_r2)

# ------------------------------------------
# GRADIENT BOOSTING REGRESSION
# ------------------------------------------
gradient_model = GradientBoostingRegressor(
    n_estimators=100,
    random_state=42
)
gradient_model.fit(X_train, y_train)
gradient_predictions = gradient_model.predict(X_test)
print("\nGradient Boosting Predictions:", gradient_predictions[:10])
print("\nGradient Boosting Actual Prices:", y_test.iloc[:10].values)

gradient_mae = mean_absolute_error(y_test,gradient_predictions)
gradient_rmse = mean_squared_error(y_test,gradient_predictions) ** 0.5
gradient_r2 = r2_score(y_test,gradient_predictions)

print("\nGRADIENT BOOSTING - EVALUATION")
print("Mean Absolute Error:", gradient_mae)
print("Root Mean Squared Error:", gradient_rmse)
print("R² Score:", gradient_r2)

# ==========================================
# 7. CROSS-VALIDATION
# ==========================================
gradient_cv_scores = cross_val_score(
    gradient_model,
    X,
    y,
    cv=5,
    scoring="r2"
)

print("\nGRADIENT BOOSTING - CROSS-VALIDATION")
print("R² Scores:", gradient_cv_scores)
print("Average R²:", gradient_cv_scores.mean())


# ==========================================
# 8. MODEL SCOREBOARD
# ==========================================
print("\nMODEL SCOREBOARD")
model_results = pd.DataFrame({
    "Model": [
        "Linear Regression",
        "Decision Tree",
        "Random Forest",
        "Gradient Boosting"
    ],
    
    "MAE": [
        linear_mae,
        tree_mae,
        forest_mae,
        gradient_mae
    ],
    
    "RMSE": [
        linear_rmse,
        tree_rmse,
        forest_rmse,
        gradient_rmse
    ],
    
    "R2": [
        linear_r2,
        tree_r2,
        forest_r2,
        gradient_r2
    ]
})

print("\nMODEL SCOREBOARD")
print(model_results)

# ==========================================
# 9. MODEL INTERPRETATION
# ==========================================
feature_importance = gradient_model.feature_importances_
print("\nNumber of Feature Importances:", len(feature_importance))
feature_importance_df = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": feature_importance
})

feature_importance_df = feature_importance_df.sort_values(
    by="Importance",
    ascending=False
)
print("\nTop 10 Most Important Features:")
print(feature_importance_df.head(10))


# ==========================================
# 3. EXPLORATORY DATA ANALYSIS
# ==========================================

# SalePrice Distribution
plt.figure(figsize=(10, 6))
plt.hist(df['SalePrice'], bins=30)
plt.xlabel('Sale Price')
plt.ylabel('Number of Houses')
plt.title('Distribution of House Sale Prices')
plt.show()

# Overall Quality vs Sale Price
plt.figure(figsize=(10, 6))
plt.scatter(df['OverallQual'], df['SalePrice'])
plt.xlabel('Overall Quality')
plt.ylabel('Sale Price')
plt.title('Overall Quality vs House Sale Price')
plt.show()

# Living Area vs Sale Price
plt.figure(figsize=(10, 6))
plt.scatter(df['GrLivArea'], df['SalePrice'])
plt.xlabel('Above Ground Living Area')
plt.ylabel('Sale Price')
plt.title('Living Area vs House Sale Price')
plt.show()

# Year Built vs Sale Price
plt.figure(figsize=(10, 6))
plt.scatter(df['YearBuilt'], df['SalePrice'])
plt.xlabel('Year Built')
plt.ylabel('Sale Price')
plt.title('Year Built vs House Sale Price')
plt.show()

