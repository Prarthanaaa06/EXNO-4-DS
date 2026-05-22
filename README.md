# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:
import numpy as np
import pandas as pd
df=pd.read_csv("bmi.csv")
df.head()
df.isnull()
df.info()
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler

X = df[['Gender', 'Height', 'Weight']]
y = df['Index']
df_std = df.copy()
scaler_std = StandardScaler()
df_std[['Height', 'Weight']] = scaler_std.fit_transform(df_std[['Height', 'Weight']])
print("\nStandard Scaled Data:")
print(df_std.head())

df_minmax = df.copy()
scaler_minmax = MinMaxScaler()
df_minmax[['Height', 'Weight']] = scaler_minmax.fit_transform(df_minmax[['Height', 'Weight']])

print("\nMin-Max Scaled Data:")
print(df_minmax.head())

df_maxabs = df.copy()
scaler_maxabs = MaxAbsScaler()
df_maxabs[['Height', 'Weight']] = scaler_maxabs.fit_transform(df_maxabs[['Height', 'Weight']])
print("\nMaxAbs Scaled Data:")
print(df_maxabs.head())

df_robust = df.copy()
scaler_robust = RobustScaler()
df_robust[['Height', 'Weight']] = scaler_robust.fit_transform(df_robust[['Height', 'Weight']])

print("\nRobust Scaled Data:")
print(df_robust.head())

df_std.to_csv("BMI_StandardScaled.csv", index=False)
df_minmax.to_csv("BMI_MinMaxScaled.csv", index=False)
df_maxabs.to_csv("BMI_MaxAbsScaled.csv", index=False)
df_robust.to_csv("BMI_RobustScaled.csv", index=False)


import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

df_std['Gender'] = df_std['Gender'].map({'Male': 1, 'Female': 0})

X_selection = df_std[['Gender', 'Height', 'Weight']]
y_selection = df_std['Index']

print("--- 1. FILTER METHOD (Correlation Matrix) ---")
plt.figure(figsize=(6, 4))
sns.heatmap(df_std.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Feature Correlation with Index")
plt.show()


print("\n--- 2. EMBEDDED METHOD (Feature Importance) ---")
model = RandomForestClassifier(random_state=42)
model.fit(X_selection, y_selection)

importances = pd.Series(model.feature_importances_, index=X_selection.columns)
importances = importances.sort_values(ascending=False)

print("Feature Importance Scores:")
print(importances)


plt.figure(figsize=(6, 3))
importances.plot(kind='barh', color='teal')
plt.title("Feature Importance via Embedded Method")
plt.xlabel("Score")
plt.gca().invert_yaxis()
plt.show()
<img width="547" height="718" alt="image" src="https://github.com/user-attachments/assets/6f2784c6-a5f0-4fbd-ae2a-35499672e292" />
<img width="630" height="551" alt="image" src="https://github.com/user-attachments/assets/418df56f-644f-4a80-86ef-f8e11e37e4bd" />
<img width="616" height="535" alt="image" src="https://github.com/user-attachments/assets/3b86da00-cff2-46ca-a292-5c9f7a58082f" />
<img width="798" height="649" alt="image" src="https://github.com/user-attachments/assets/84ce84d6-bb2c-426f-9008-1c6632a5669e" />
<img width="824" height="405" alt="image" src="https://github.com/user-attachments/assets/c157bde3-8a21-4152-9909-cd431347ec2b" />


# RESULT:
All four scaling techniques successfully mapped the numerical features
