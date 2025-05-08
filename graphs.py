import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df  = pd.read_csv('/content/sample_data/california_housing_test.csv')
print(df.head())

sns.set(style="whitegrid")

df['age_bucket'] = pd.cut(df['housing_median_age'], bins=[0, 20, 30, 40, 50, 60], labels=['0-20','21-30','31-40','41-50','51-60'])
bar_data=df.groupby('age_bucket')['median_income'].mean().reset_index()
plt.figure(figsize=(8, 5))
sns.barplot(data=bar_data, x='age_bucket', y='median_income', palette='mako')
plt.show()

sns.histplot(df['total_bedrooms'], bins=10, kde=True, color='teal')
plt.figure(figsize=(8, 5))
plt.title('Income Distribution')
plt.show()

plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='total_rooms', y='median_house_value', hue='median_income', palette='coolwarm', size='median_house_value', sizes=(20, 200))
plt.title('House Values by Location')
plt.show()