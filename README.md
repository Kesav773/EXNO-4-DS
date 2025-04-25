# EXNO:4-Feature Scaling and Selection
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
import pandas as pd 
from scipy import stats 
import numpy as np
df=pd.read_csv("/content/bmi.csv")
df.head()

![Screenshot 2025-04-25 110906](https://github.com/user-attachments/assets/260f501b-6379-4def-9f71-988fea351812)

 df_null_sum=df.isnull().sum()
 df_null_sum

![Screenshot 2025-04-25 110911](https://github.com/user-attachments/assets/be217cbb-f5a9-4ce7-9916-7ae7cfd132e1)

 df.dropna()

![Screenshot 2025-04-25 110918](https://github.com/user-attachments/assets/360e7918-3b4f-44e3-9fca-518bbe1afc57)

 max_vals = np.max(np.abs(df[['Height', 'Weight']]), axis=0)
 max_vals

![Screenshot 2025-04-25 110928](https://github.com/user-attachments/assets/f8158a2f-ef64-4a58-84fb-ae38c0591a80)

from sklearn.preprocessing import StandardScaler
df1=pd.read_csv("/content/bmi.csv")
df1.head()

![Screenshot 2025-04-25 110935](https://github.com/user-attachments/assets/efac3539-d727-4eed-8872-173712efddcc)

sc=StandardScaler()
df1[['Height','Weight']]=sc.fit_transform(df1[['Height','Weight']])
df1.head(10)

![Screenshot 2025-04-25 110942](https://github.com/user-attachments/assets/97c2a055-8011-4305-b4b5-465e3a95e118)

 from sklearn.preprocessing import MinMaxScaler
 scaler=MinMaxScaler()
 df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
 df.head(10)

![Screenshot 2025-04-25 110956](https://github.com/user-attachments/assets/6ef98765-7eb3-4726-98ca-8189afe592d3)

 from sklearn.preprocessing import MaxAbsScaler
 scaler = MaxAbsScaler()
 df3=pd.read_csv("/content/bmi.csv")
 df3.head()

![Screenshot 2025-04-25 111002](https://github.com/user-attachments/assets/950cc67a-1151-4bf5-b648-879492f2c239)

 df3[['Height','Weight']]=scaler.fit_transform(df3[['Height','Weight']])
 df3

![Screenshot 2025-04-25 111011](https://github.com/user-attachments/assets/ea231069-98ee-4dd4-8ad7-d47dbe35e90b)

 from sklearn.preprocessing import RobustScaler
 scaler = RobustScaler()
 df4=pd.read_csv("/content/bmi.csv")
 df4.head()

![Screenshot 2025-04-25 111017](https://github.com/user-attachments/assets/5a02c53b-fa89-4f74-85e9-cbfb84cdd775)

 df4[['Height','Weight']]=scaler.fit_transform(df4[['Height','Weight']])
 df4.head()

![Screenshot 2025-04-25 111023](https://github.com/user-attachments/assets/48caebe4-53cc-4ce0-8592-cdaf6a4da69a)

 df=pd.read_csv("/content/income(1) (1).csv")
 df.info()

![Screenshot 2025-04-25 111030](https://github.com/user-attachments/assets/57e1c9ed-b7b6-4039-8453-cd46345e7735)

df

![Screenshot 2025-04-25 111053](https://github.com/user-attachments/assets/2d26f1ba-6ceb-4761-ad60-579d83b1c5bf)

df.info()

![Screenshot 2025-04-25 111109](https://github.com/user-attachments/assets/26504d7d-ae2e-4c40-b946-296b429a568c)

df_null_sum=df.isnull().sum()
df_null_sum

![Screenshot 2025-04-25 111118](https://github.com/user-attachments/assets/8a91e556-8376-47c5-9019-6b6c7070f23c)

categorical_columns = ['JobType', 'EdType', 'maritalstatus', 'occupation', 'relationship', 'race', 'gender', 'nativecountry']
df[categorical_columns] = df[categorical_columns].astype('category')
df[categorical_columns]

![Screenshot 2025-04-25 111126](https://github.com/user-attachments/assets/4211acff-ebc9-4018-ade6-f285db35deb6)

df[categorical_columns] = df[categorical_columns].apply(lambda x: x.cat.codes)
df[categorical_columns]

![Screenshot 2025-04-25 111134](https://github.com/user-attachments/assets/660f4df8-ab07-4e8c-9e0e-38270071ae89)

 import pandas as pd
 from sklearn.feature_selection import SelectKBest, chi2, f_classif
 categorical_columns = ['JobType', 'EdType', 'maritalstatus', 'occupation', 'relationship', 'race', 'gender', 'nativecountry']
 df[categorical_columns] = df[categorical_columns].astype('category')
 df[categorical_columns]

![Screenshot 2025-04-25 111141](https://github.com/user-attachments/assets/20bcc0b1-610d-4605-b890-2311882a6bb5)

df[categorical_columns] = df[categorical_columns].apply(lambda x: x.cat.codes)
df[categorical_columns]

![Screenshot 2025-04-25 111152](https://github.com/user-attachments/assets/ce55530c-419e-4284-a095-92e90172dec3)

import pandas as pd
 from sklearn.feature_selection import RFE
 from sklearn.linear_model import LogisticRegression
 categorical_columns = ['JobType', 'EdType', 'maritalstatus', 'occupation', 'relationship', 'race', 'gender', 'nativecountry']
 df[categorical_columns] = df[categorical_columns].astype('category')
 df[categorical_columns]

![Screenshot 2025-04-25 111215](https://github.com/user-attachments/assets/c0191f7b-2187-4b95-9c64-2193b4233bc3)

df[categorical_columns] = df[categorical_columns].apply(lambda x: x.cat.codes)
df[categorical_columns]

![Screenshot 2025-04-25 111224](https://github.com/user-attachments/assets/8bbb644d-0769-4f66-bef9-52fc641666f6)

 k_anova = 5
 selector_anova = SelectKBest(score_func=f_classif, k=k_anova)
 X_anova = selector_anova.fit_transform(X, y)
 selected_features_anova = X.columns[selector_anova.get_support()]
 print("\nSelected features using ANOVA:")
 print(selected_features_anova)

![Screenshot 2025-04-25 113030](https://github.com/user-attachments/assets/04b5f2ef-dcfb-4af9-9ed1-a57812885e3c)

 X = df.drop(columns=['SalStat'])
 y = df['SalStat']
 logreg = LogisticRegression()
 n_features_to_select = 6
 rfe = RFE(estimator=logreg, n_features_to_select=n_features_to_select)
 rfe.fit(X, y)

![Screenshot 2025-04-25 113046](https://github.com/user-attachments/assets/fa162a2b-041d-442d-a115-ea2f45852338)

 selected_features = X.columns[rfe.support_]
 print("Selected features using RFE:")
 print(selected_features)

![Screenshot 2025-04-25 113053](https://github.com/user-attachments/assets/48fdaaaa-2bb5-4526-8be3-298fb74a4ca4)

 from sklearn.model_selection import train_test_split
 from sklearn.metrics import accuracy_score
 from sklearn.ensemble import RandomForestClassifier
 X_selected = X[selected_features]
 X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.3, random_state=42)
 rf = RandomForestClassifier(n_estimators=100, random_state=42)
 rf.fit(X_train, y_train)
 y_pred = rf.predict(X_test)
 accuracy = accuracy_score(y_test, y_pred)
 print(f"Model accuracy using Fisher Score selected features: {accuracy}")

![Screenshot 2025-04-25 113102](https://github.com/user-attachments/assets/3676588a-69ae-49af-aae8-fccb0816d4f3)


 








# RESULT:
Thus, feature selection and feature scaling was implemented successfuly

