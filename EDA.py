import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

auto=pd.read_csv("Car_Insurance_Claim - Copy.csv")

# for col in auto.columns:
#     print('{} : {}'.format(col,auto[col].unique()))

# for col in auto.columns:
#     auto[col].replace({'?':np.nan},inplace=True)

# print(auto.info())
auto.drop('ANNUAL_MILEAGE', inplace=True, axis=1)
# print(auto.isnull().sum() / auto.shape[0] * 100)
num_col = ['CREDIT_SCORE']
for col in num_col:
    auto[col]=pd.to_numeric(auto[col])
    auto[col].fillna(auto[col].mean(), inplace=True)
# print(auto.head())

# print(auto.isnull().sum() / auto.shape[0] * 100)
# sns.heatmap(auto.isnull(),cbar=False,cmap='viridis')

# plt.show()


#heatmap
plt.figure(figsize=(10,10))
sns.heatmap(auto.corr(),annot=True,cmap='coolwarm')
plt.show()

# check if data is skewed
# sns.displot(auto, x="INCOME")
# plt.show()
