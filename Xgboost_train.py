#Necessary libraries
import pandas as pd
import numpy as np
import pickle

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, PolynomialFeatures

from xgboost import XGBClassifier


#Loading the Dataset
df = pd.read_csv("diabetes_dataset.csv")

#Seperating the Dataset
X = df.drop('Outcome',axis=1)
y = df['Outcome']


numeric_features = X.select_dtypes(include = ['int64','float64']).columns

#Train Test Split 
X_train, X_test, y_train, y_test= train_test_split(X, y, test_size = 0.2, stratify=y, random_state=42)

#Handling Zeros
cols_with_zeros = ["Glucose","BloodPressure","SkinThickness","Insulin","BMI"]
X_train[cols_with_zeros] = X_train[cols_with_zeros].replace(0, np.nan)
X_test[cols_with_zeros] = X_test[cols_with_zeros].replace(0, np.nan)

imputer = SimpleImputer(strategy='median')
X_train_imputed = pd.DataFrame( imputer.fit_transform(X_train), columns=X_train.columns, index=X_train.index )
X_test_imputed = pd.DataFrame( imputer.transform(X_test), columns=X_test.columns, index=X_test.index )


#Handling Outliers
numeric_outlier_cols = [
    "Pregnancies",
    "SkinThickness",
    "DiabetesPedigreeFunction",
]

# For X_train
X_train_capped_without_encoding = X_train_imputed.copy()

for col in numeric_outlier_cols:
    Q1 = X_train_capped_without_encoding[col].quantile(0.25)
    Q3 = X_train_capped_without_encoding[col].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    outliers = X_train_capped_without_encoding[(X_train_capped_without_encoding[col] < lower) | (X_train_capped_without_encoding[col] > upper)]
    X_train_capped_without_encoding[col] = X_train_capped_without_encoding[col].clip(lower, upper)

# For X_test
X_test_capped_without_encoding = X_test_imputed.copy()

for col in numeric_outlier_cols:
    Q1 = X_test_capped_without_encoding[col].quantile(0.25)
    Q3 = X_test_capped_without_encoding[col].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    outliers = X_test_capped_without_encoding[(X_test_capped_without_encoding[col] < lower) | (X_test_capped_without_encoding[col] > upper)]
    X_test_capped_without_encoding[col] = X_test_capped_without_encoding[col].clip(lower, upper)

# Reassigning of X_train, X_test after preprocessing
X_train = X_train_capped_without_encoding
X_test = X_test_capped_without_encoding

#Model Pipeline
XGBClassifier_pipeline = Pipeline([
    ('poly', PolynomialFeatures(degree=1, include_bias=False)),
    ('scaler', RobustScaler()),
    ('clf', XGBClassifier(
        random_state=42,
        eval_metric="logloss",
        n_estimators=100,
        subsample=1,
        reg_lambda=2.0,
        max_depth=3,
        learning_rate=0.05,
        colsample_bytree=1.0
    ))
])

#Fitting the model
XGBClassifier_pipeline.fit(X_train, y_train)

# Getting pickle file
filename = "Diabetes_model.pkl"

with open(filename, "wb") as file:
    pickle.dump(XGBClassifier_pipeline, file)