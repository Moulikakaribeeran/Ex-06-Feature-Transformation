# Ex-06-Feature-Transformation
AIM:

   To read the given data and perform Feature Transformation process and save the data to the file.
   
EXPLAINATION:

  Feature Transformation is a technique by which we can boost our model performance. Feature transformation is a mathematical transformation in which we apply a mathematical formula to a particular column(feature) and transform the values which are useful for our further analysis.

ALGORITHM:

 STEP 1 Read the given Data

 STEP 2 Clean the Data Set using Data Cleaning Process

 STEP 3 Apply Feature Transformation techniques to all the features of the data set

 STEP 4 Save the data to the file
 
 CODE AND OUTPUT:
 
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import statsmodels.api as sm

import scipy.stats as stats

from sklearn.preprocessing import QuantileTransformer

df=pd.read_csv("C:\\Users\\lenovo\\Documents\\Data_to_Transform.csv")

df

![1](https://user-images.githubusercontent.com/95409048/197516426-5ac55490-472e-4e73-aa91-f4a00f3bb6d5.png)

df.head()

![2](https://user-images.githubusercontent.com/95409048/197516762-ed7f12c0-ae82-4a21-9d57-f891ef63c984.png)

df.isnull().sum()

![3](https://user-images.githubusercontent.com/95409048/197517281-8f981138-5b66-4f80-b425-dbd088a8a909.png)

df.info()

![4](https://user-images.githubusercontent.com/95409048/197517502-31680bb8-e760-493e-a895-05c85e308ab2.png)

df.describe()

![5](https://user-images.githubusercontent.com/95409048/197517727-f2253b01-1f8d-43a4-bbfb-51d42f443edf.png)

df1 = df.copy()

sm.qqplot(df1.HighlyPositiveSkew,fit=True,line='45')

plt.show()

![6](https://user-images.githubusercontent.com/95409048/197518006-93da55c1-9b41-4ecb-913b-61c3b5571e3e.png)

sm.qqplot(df1.HighlyNegativeSkew,fit=True,line='45')

plt.show()

![7](https://user-images.githubusercontent.com/95409048/197518247-d48abbb9-7845-4dbf-b33b-7bba96d20715.png)

sm.qqplot(df1.ModeratePositivSkew,fit=True,line='45')

plt.show()

![8](https://user-images.githubusercontent.com/95409048/197518395-c8aec216-2503-46eb-b9f8-6abe10ed3561.png)

sm.qqplot(df1.ModerateNegativeSkew,fit=True,line='45')

plt.show()

![9](https://user-images.githubusercontent.com/95409048/197518938-f100f7dc-ff0b-469c-b836-b78c93209a6d.png)

df1['HighlyPositiveSkew'] = np.log(df1.HighlyPositiveSkew)

sm.qqplot(df1.HighlyPositiveSkew,fit=True,line='45')

plt.show()

![10](https://user-images.githubusercontent.com/95409048/197519027-8711489c-37b4-42c2-907c-9a38eac7c38c.png)

df2 = df.copy()

df2['HighlyPositiveSkew'] = 1/df2.HighlyPositiveSkew

sm.qqplot(df2.HighlyPositiveSkew,fit=True,line='45')

plt.show()

![11](https://user-images.githubusercontent.com/95409048/197519251-967b899e-5627-4ccb-98d2-23d9ec18303d.png)

df3 = df.copy()

df3['HighlyPositiveSkew'] = df3.HighlyPositiveSkew**(1/1.2)

sm.qqplot(df2.HighlyPositiveSkew,fit=True,line='45')

plt.show()

![12](https://user-images.githubusercontent.com/95409048/197519477-37646e26-b030-4aa4-888e-b988220b3e99.png)

df4 = df.copy()

df4['ModeratePositiveSkew_1'],parameters =stats.yeojohnson(df4.ModeratePositivSkew)

sm.qqplot(df4.ModeratePositivSkew,fit=True,line='45')

plt.show()

![13](https://user-images.githubusercontent.com/95409048/197519662-38b54363-9abb-4dda-936d-02a37cd93a51.png)

from sklearn.preprocessing import PowerTransformer 

trans = PowerTransformer("yeo-johnson")

df5 = df.copy()

df5['ModerateNegativeSkew_1'] = pd.DataFrame(trans.fit_transform(df5[['ModerateNegativeSkew']]))

sm.qqplot(df5['ModerateNegativeSkew_1'],line='45')

plt.show()

![14](https://user-images.githubusercontent.com/95409048/197519922-deec639c-2687-4274-9b5e-84ddbc7c2bef.png)

from sklearn.preprocessing import QuantileTransformer

qt = QuantileTransformer(output_distribution = 'normal')

df5['ModerateNegativeSkew_2'] = pd.DataFrame(qt.fit_transform(df5[['ModerateNegativeSkew']]))

sm.qqplot(df5['ModerateNegativeSkew_2'],line='45')

plt.show()

![15](https://user-images.githubusercontent.com/95409048/197520150-d5e514bb-98bc-45e9-a8f8-55ba3f401bb2.png)

RESULT:
    
    Thus the Feature Transformation for the given datasets had been executed successfully.

 
 
