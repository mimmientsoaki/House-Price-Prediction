import pandas as pd
from sklearn.model_selection import train_test_split

#Data Loading
df = pd.read_csv('train.csv')
print(df.head())
print("\nDataset Shape", df.shape)
print("\nColumn Names:", df.columns.to_list())
print("\nDataset Info:", df.info())
print("\nMissing Values:\n", df.isnull().sum())
print("\nStatistical Summary:\n", df.describe())

#Data Preprocessing/ Cleaning
#Handeling Missing Values
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

#Encoding Categorical Variables
categorical_columns = df.select_dtypes(
    include="object"
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
print(df.select_dtypes(include="object").columns.tolist())
print("\nData Types:",df.dtypes.value_counts())

#FEATURE & TARGET SEPARATION
X = df.drop("SalePrice", axis=1)
y = df["SalePrice"]

print("\nData Shape:",X.shape)
print("Target Shape:", y.shape)

#TRAIN / TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print("\nTrain Data Shape:", X_train.shape)
print("Test Data Shape:", X_test.shape)
print("Train Target Shape:", y_train.shape)
print("Test Target Shape:", y_test.shape)