"""
Titanic Dataset Preprocessing for Logistic Regression
--------------------------------------------------------
Prepares the raw Kaggle Titanic dataset for a logistic regression
model predicting Survived (0/1).
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from helper_functions_2 import *
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

# ---------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------
df = pd.read_csv('Titanic-Dataset.csv')
print(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns")

# ---------------------------------------------------------
# 2. Feature engineering we need Name and Cabin to build new features)
# ---------------------------------------------------------

# 2a. Title extracted from Name (Mr, Mrs, Miss, Master, Rare...)
df['Title'] = df['Name'].str.extract(r',\s*([^.]*)\.')
title_map = {
    'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs',
    'Lady': 'Rare', 'Countess': 'Rare', 'Capt': 'Rare', 'Col': 'Rare',
    'Don': 'Rare', 'Dr': 'Rare', 'Major': 'Rare', 'Rev': 'Rare',
    'Sir': 'Rare', 'Jonkheer': 'Rare', 'Dona': 'Rare'
}
df['Title'] = df['Title'].replace(title_map)
# keeping only the common categories
common_titles = ['Mr', 'Miss', 'Mrs', 'Master']
df['Title'] = df['Title'].apply(lambda t: t if t in common_titles else 'Rare')

# 2b. Family size + IsAlone flag
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

# 2c. HasCabin flag 
df['HasCabin'] = df['Cabin'].notna().astype(int)

# ---------------------------------------------------------
# 3. Handle missing values
# ---------------------------------------------------------

# 3a. Age: impute using median Age within each Pclass group
#     (better than a single global median since Age clearly
#     varies by class: 37 / 29 / 24 for class 1/2/3)
df['Age'] = df.groupby('Pclass')['Age'].transform(lambda x: x.fillna(x.median()))

# 3b. Embarked: filling missing value with the mode ('S')
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

# 3c. Fare: no missing values in this dataset, but guard anyway
df['Fare'] = df['Fare'].fillna(df['Fare'].median())

assert df[['Age', 'Embarked', 'Fare']].isnull().sum().sum() == 0, "Missing values remain!"

# ---------------------------------------------------------
# 4. Drop columns that don't help a logistic regression model
# ---------------------------------------------------------
# PassengerId: just a row index, no predictive signal
# Name, Ticket: raw free text, no direct numeric meaning (we
#   already extracted Title, which captures the useful signal)
# Cabin: 77% missing, and we've captured HasCabin already
# SibSp, Parch: dropped because FamilySize = SibSp + Parch + 1
#   is a perfect linear combination of them (infinite VIF)
# Sex: dropped because it's almost fully redetermined by the
#   Title_* dummies (VIF > 90), so Title alone carries that signal
df = df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin',
                       'SibSp', 'Parch', 'Sex'])

# ---------------------------------------------------------
# 5. Encode categorical variables
# ----------------------------------------

# Embarked, Title: one-hot encode (drop_first avoids the
# dummy-variable trap / perfect multicollinearity)
df = pd.get_dummies(df, columns=['Embarked', 'Title'], drop_first=True)

# convert any bool columns from get_dummies into int (0/1)
bool_cols = df.select_dtypes(include='bool').columns
df[bool_cols] = df[bool_cols].astype(int)

print("\nColumns after encoding:")
print(df.columns.tolist())

# ---------------------------------------------------------
# 6. Train/test split
# ---------------------------------------------------------
X = df.drop(columns=['Survived'])
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain size: {X_train.shape}, Test size: {X_test.shape}")

# ---------------------------------------------------------
# 7. Feature scaling
# ---------------------------------------------------------
# Logistic regression (especially with regularization) benefits
# from scaled features. Fit scaler on TRAIN ONLY, then apply to
# both train and test to avoid data leakage.
numeric_cols = ['Age', 'Fare', 'FamilySize']

scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()
X_train_scaled[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
X_test_scaled[numeric_cols] = scaler.transform(X_test[numeric_cols])

print("\nFinal feature set:")
print(X_train_scaled.columns.tolist())
print("\nSample of processed training data:")
print(X_train_scaled.head())


# ===========================================================
# 8. TRAIN LOGISTIC REGRESSION using helper_functions_2
# ===========================================================

# helper_functions_2's functions expect raw numpy arrays, not
# DataFrames — convert here. Also flatten y to shape (m,)
X_train_np = X_train_scaled.to_numpy()
X_test_np = X_test_scaled.to_numpy()
y_train_np = y_train.to_numpy().flatten()
y_test_np = y_test.to_numpy().flatten()

m, n = X_train_np.shape

# 8a. Initialize parameters
np.random.seed(1)
w_init = np.random.randn(n) * 0.01   # small random values, not zeros
b_init = 0.0

# 8b. Run regularized gradient descent
alpha = 0.01
num_iters = 10000
lambda_ = 1.0

w, b, J_history, w_history = gradient_descent(
    X_train_np, y_train_np, w_init, b_init,
    compute_cost_reg, compute_gradient_reg,
    alpha, num_iters, lambda_
)

print(f"\nFinal cost: {J_history[-1]:.4f}")

# 8c. Evaluate
train_preds = predict(X_train_np, w, b)
test_preds = predict(X_test_np, w, b)

train_acc = np.mean(train_preds == y_train_np) * 100
test_acc = np.mean(test_preds == y_test_np) * 100

print(f"\nTrain accuracy: {train_acc:.2f}%")
print(f"Test accuracy:  {test_acc:.2f}%")

# 8d. Feature weight inspection
feature_names = X_train_scaled.columns.tolist()
coef_df = pd.DataFrame({'feature': feature_names, 'weight': w})
coef_df = coef_df.reindex(coef_df.weight.abs().sort_values(ascending=False).index)
print("\nFeature weights (sorted by magnitude):")
print(coef_df.to_string(index=False))

# ===========================================================
# 9. TRAIN LOGISTIC REGRESSION using sklearn
# ===========================================================
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix,
    classification_report
)

# 9a. Fit model
sk_model = LogisticRegression(max_iter=1000, random_state=42)
sk_model.fit(X_train_scaled, y_train)

# 9b. Predict
sk_train_preds = sk_model.predict(X_train_scaled)
sk_test_preds = sk_model.predict(X_test_scaled)

# 9c. Evaluate
sk_train_acc = accuracy_score(y_train, sk_train_preds) * 100
sk_test_acc = accuracy_score(y_test, sk_test_preds) * 100

print(f"\n[sklearn] Train accuracy: {sk_train_acc:.2f}%")
print(f"[sklearn] Test accuracy:  {sk_test_acc:.2f}%")

print("\n[sklearn] Confusion matrix (test set):")
print(confusion_matrix(y_test, sk_test_preds))

print("\n[sklearn] Classification report (test set):")
print(classification_report(y_test, sk_test_preds, target_names=['Did not survive', 'Survived']))

# 9d. Coefficients (comparable to your from-scratch weights)
sk_coef_df = pd.DataFrame({
    'feature': X_train_scaled.columns,
    'weight': sk_model.coef_[0]
})
sk_coef_df = sk_coef_df.reindex(sk_coef_df.weight.abs().sort_values(ascending=False).index)
print("\n[sklearn] Feature weights (sorted by magnitude):")
print(sk_coef_df.to_string(index=False))
print(f"\n[sklearn] Intercept (b): {sk_model.intercept_[0]:.4f}")


X_vif = add_constant(X_train_scaled)
vif_df = pd.DataFrame()
vif_df['feature'] = X_vif.columns
vif_df['VIF'] = [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])]
vif_df = vif_df[vif_df['feature'] != 'const'].sort_values('VIF', ascending=False)
print(vif_df.to_string(index=False))