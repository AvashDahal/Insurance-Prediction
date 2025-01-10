import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib.pyplot import xticks
from matplotlib.pyplot import yticks
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle as pk

df = pd.read_csv("./insurance.csv")
print(df)
print(df.describe())
df.info()
row,col=df.shape
print(row)
print(col)
print(df.isnull().sum())
print(df.isna().sum())
print(df.columns)
print(df.info())
df1=df[['age','sex','bmi','children','smoker','region','expenses']]
new_df=df[['age','bmi','children','expenses']]
print(new_df.corr())
# plt.figure(figsize=(15,5))
# sns.heatmap(new_df.corr(),annot=True)
# plt.show()
gender=pd.get_dummies(df1['sex'],drop_first=True)
print(gender)
df1=pd.concat([df1,gender],axis=1)
print(df1)
df1['smoker'] = df1['smoker'].replace({'yes': True, 'no': False})
print(df1)
df1=df1.drop(['sex'],axis=1)
print(df1)
region = pd.get_dummies(df1['region'],drop_first=True)
print(region)
df1=pd.concat([df1,region],axis=1)
print(df1)
df1=df1.drop(['region'],axis=1)
print(df1)
print(df1.corr())
# plt.figure(figsize=(15,5))
# sns.heatmap(df1.corr(),annot=True)
# plt.show()
df_train, df_test = train_test_split(df1, train_size = 0.85, test_size = 0.15, random_state = 1)
print(df_train)
print(df_train.shape)
print(df_test.shape)
X_train=df_train[['age' ,'bmi','children','smoker','male','northwest','southeast','southwest']]
Y_train=df_train['expenses']
lr = LinearRegression()
lr_model = lr.fit(X_train, Y_train)
filename = 'model.pickle'
pk.dump(lr_model, open(filename, 'wb'))
data = df1.iloc[100:101]
data
actual_expense=data['expenses']
data=data.drop(['expenses'],axis=1)
print(data)
print("Predicted Salary",lr_model.predict(data))
print("Actual Salary",actual_expense)
data = {
    'age': [int(input("Enter your age\n"))],
    'bmi': [float(input("Enter your BMI\n"))],
    'children': [int(input("Enter number of children\n"))],
    'smoker': [int(input("Enter 1 if you are a smoker, 0 if not\n"))],
    'male': [int(input("Enter 1 if you are male, 0 if female\n"))],
    'northwest': [int(input("Enter 1 if you live in the northwest, 0 if not\n"))],
    'southeast': [int(input("Enter 1 if you live in the southeast, 0 if not\n"))],
    'southwest': [int(input("Enter 1 if you live in the southwest, 0 if not\n"))]
}
df = pd.DataFrame(data)
predicted_expenses= lr_model.predict(df)
print("predicted expense",np.round(predicted_expenses,2))
print(predicted_expenses)
df
plt.scatter(Y_train,lr_model.predict(X_train),color = 'red')
plt.show()
print("Linear regression = ",round(lr_model.score(X_train, Y_train)*100))