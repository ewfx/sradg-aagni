import pandas as pd
from sklearn.preprocessing import *
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss
from sklearn.preprocessing import StandardScaler
# !pip install ppscore
# import ppscore as pps
import statsmodels.api as sm
from scipy.stats import probplot
from sklearn.metrics import mean_absolute_error
#%matplotlib inline
#!pip install PyCustom
#import PyCustom
from statistics import mean, stdev
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, confusion_matrix
import pickle
import os
print(os.getcwd())
file_path = os.getcwd().replace("\\", "/")
df = pd.read_csv(file_path+"/code/src/LoanParams.csv")

sns.set(style="whitegrid")
# plt.style.use('white')
plt.rcParams.update({"grid.linewidth":0.5, "grid.alpha":0.5})
colormap = sns.diverging_palette(220, 10, as_cmap=True)

# Heatmap for null/missing values
sns.heatmap(df.isnull(),cbar=False,yticklabels=False,cmap = 'viridis')

#print(df["REASON"].value_counts())
#print(df["JOB"].value_counts())

# We can fill the missing values with the mode, i.e. "Other", or we can fill the missing values depending on the distribution of the non-null values. 
df["REASON"]=df["REASON"].fillna(value="Other reason")
df["JOB"]=df["JOB"].fillna(value=0)
df["DEROG"]=df["DEROG"].fillna(value=0)        # Filling the missing value with the mode
df["DELINQ"]=df["DELINQ"].fillna(value=0)    # Filling the missing value with the mode
df["MORTDUE"]=df["MORTDUE"].fillna(value=df["MORTDUE"].mode()[0])
df["VALUE"]=df["VALUE"].fillna(value=df["VALUE"].mode()[0])
df["YOJ"]=df["YOJ"].fillna(value=df["YOJ"].mode()[0])
df["CLAGE"]=df["CLAGE"].fillna(value=df["CLAGE"].mode()[0])
df["NINQ"]=df["NINQ"].fillna(value=df["NINQ"].mode()[0])
df["CLNO"]=df["CLNO"].fillna(value=df["CLNO"].mode()[0])
df["DEBTINC"]=df["DEBTINC"].fillna(value=df["DEBTINC"].mode()[0])
# print(df["JOB"].isna().sum())

# label_encoder = preprocessing.LabelEncoder()
# df['JOB']= label_encoder.fit_transform(df['JOB'])
# df['REASON']= label_encoder.fit_transform(df['REASON'])
df = df.join(pd.get_dummies(df["JOB"]))
df = df.join(pd.get_dummies(df["REASON"]))
df.drop(["JOB","REASON"],axis=1,inplace=True)


# Define target variable and features
X = df.drop('BAD', axis=1)  # Features (all columns except 'BAD')
y = df['BAD']  # Target variable
X.columns = X.columns.astype(str)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
# Initialize the Random Forest model
rf = RandomForestClassifier(random_state=42)

# Train the model
rf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rf.predict(X_test)

# Evaluate the model performance
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

#Save the trained model and the scaler
with open('EvaluateRiskModel.pkl', 'wb') as model_file:
    pickle.dump(rf, model_file)

with open('EvaluateRiskModelScaler.pkl', 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)
