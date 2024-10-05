## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:
~~~
import pandas as pd
df=pd.read_csv("/content/Encoding Data.csv")
df
~~~
![Screenshot 2024-10-05 133106](https://github.com/user-attachments/assets/1377eb96-6943-491d-8b6d-6b6b4177f8f8)
~~~
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
pm=['Hot','Warm','Cold']
e1=OrdinalEncoder(categories=[pm])
e1.fit_transform(df[["ord_2"]])
~~~
![Screenshot 2024-10-05 133118](https://github.com/user-attachments/assets/3bf7b73a-ce18-40e7-9229-22b239b8f3fc)
~~~
df['bo2']=e1.fit_transform(df[["ord_2"]])
df
~~~
![Screenshot 2024-10-05 133127](https://github.com/user-attachments/assets/b167d43f-1d24-4fe6-ad58-11c7930e1998)
~~~
le=LabelEncoder()
dfc=df.copy()
dfc['ord_2']=le.fit_transform(dfc['ord_2'])
dfc
~~~
![Screenshot 2024-10-05 133147](https://github.com/user-attachments/assets/bf9ecf7a-28c6-412c-befd-025bb665157c)
~~~
from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(sparse=False)
df2=df.copy()
enc=pd.DataFrame(ohe.fit_transform(df2[["nom_0"]]))
~~~
![Screenshot 2024-10-05 133202](https://github.com/user-attachments/assets/ed41ec82-a41a-40d0-bb55-8fd060ca83f2)
~~~
df2=pd.concat([df2,enc],axis=1)
df2
~~~
![Screenshot 2024-10-05 133211](https://github.com/user-attachments/assets/d1dfe5ab-8ebb-41d1-807f-32db5fc28adc)
~~~
pd.get_dummies(df2,columns=["nom_0"])
~~~
![Screenshot 2024-10-05 133218](https://github.com/user-attachments/assets/4e0cfd8c-6da5-46c8-863b-c2e499112bf2)
~~~
pip install --upgrade category_encoders
~~~
![Screenshot 2024-10-05 133228](https://github.com/user-attachments/assets/dd515055-eef8-4194-8c21-56536f20f1df)
~~~
from category_encoders import BinaryEncoder
df=pd.read_csv("/content/data.csv")
df
~~~
![Screenshot 2024-10-05 133236](https://github.com/user-attachments/assets/86230024-dfee-4a11-b3dc-cf65d7ec1c9c)
~~~
be=BinaryEncoder()
nd=be.fit_transform(df['Ord_2'])
dfb=pd.concat([df,nd],axis=1)
dfb1=df.copy()
dfb
~~~
![Screenshot 2024-10-05 133246](https://github.com/user-attachments/assets/275ac429-98e9-4698-96db-854106e3ec8d)
~~~
from category_encoders import TargetEncoder
te=TargetEncoder()
CC=df.copy()
new=te.fit_transform(X=CC["City"],y=CC["Target"])
CC=pd.concat([CC,new],axis=1)
CC
~~~
![Screenshot 2024-10-05 133253](https://github.com/user-attachments/assets/10c65924-523b-4def-a5be-57937e1c9f26)
~~~
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("/content/Data_to_Transform.csv")
df
~~~
![Screenshot 2024-10-05 133305](https://github.com/user-attachments/assets/372d278d-755b-4bff-b6e5-fb985f3a00d3)
~~~
df.skew()
~~~
![Screenshot 2024-10-05 133312](https://github.com/user-attachments/assets/ed050a7c-9b50-4bb5-a14e-6e24f3976928)
~~~
np.log(df["Highly Positive Skew"])
~~~
![Screenshot 2024-10-05 133322](https://github.com/user-attachments/assets/02b5644c-6d4d-44f5-b2a2-eed36dab4556)
~~~
np.reciprocal(df["Moderate Positive Skew"])
~~~
![Screenshot 2024-10-05 133331](https://github.com/user-attachments/assets/28c0db49-f69b-47b6-9420-64611e62b1af)
~~~
np.sqrt(df["Highly Positive Skew"])
~~~
![Screenshot 2024-10-05 133352](https://github.com/user-attachments/assets/dfb812ed-00ac-4304-b0e1-9f2c9f059411)
~~~
np.square(df["Highly Positive Skew"])
~~~
![Screenshot 2024-10-05 133357](https://github.com/user-attachments/assets/031c10e6-e697-4942-9284-c3e4f5a16e6e)
~~~
df["Highly Positive Skew_boxcox"], parameters=stats.boxcox(df["Highly Positive Skew"])
df
~~~
![Screenshot 2024-10-05 133408](https://github.com/user-attachments/assets/c944c9a1-ef5b-4a60-9483-550678f3fa8b)
~~~
df["Moderate Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Moderate Negative Skew"])
~~~
![Screenshot 2024-10-05 133446](https://github.com/user-attachments/assets/dc524df7-8c4d-4771-93ef-88c86701c585)
~~~
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
~~~
![Screenshot 2024-10-05 133455](https://github.com/user-attachments/assets/897838a6-75d2-42f6-a08b-1fe38fca70d5)
~~~
sm.qqplot(np.reciprocal(df["Moderate Negative Skew_1"]),line='45')
plt.show()
~~~
![Screenshot 2024-10-05 133502](https://github.com/user-attachments/assets/0b43b6db-9401-4a78-8d73-f59e95d0d509)
~~~
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)

df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])

sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
~~~
![Screenshot 2024-10-05 133513](https://github.com/user-attachments/assets/e8c3d04e-7ac6-4041-b3ae-de6c4bec118e)
~~~
df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])
sm.qqplot(df["Highly Negative Skew"],line='45')
plt.show()
~~~
![Screenshot 2024-10-05 133518](https://github.com/user-attachments/assets/6c469376-a76d-4e7d-836a-5d82545f0a58)
~~~
sm.qqplot(df["Highly Negative Skew_1"],line='45')
plt.show()
~~~
![Screenshot 2024-10-05 133526](https://github.com/user-attachments/assets/468932ea-6bc8-4305-adb1-d513da406455)
~~~
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
plt.show()
~~~
![Screenshot 2024-10-05 133532](https://github.com/user-attachments/assets/3d42dbcb-dc67-4b00-999c-c04199b80632)


# RESULT:
       Thus the given data, Feature Encoding, Transformation process and save the data to a file was performed successfully.



       
