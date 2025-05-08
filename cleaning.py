import pandas as pd
from google.colab import files

house_data = {
    "age": [30, 25, 50, 32, 42, 40, 45, 43, None, 32, 60, 50, 47, -2, 48, 36, 36, 28, None, 29, 30, 54],
    "owns": [True, True, True, False, None, True, True, False, None, None, True, True, False, True, False, True, True, True, False, True, True, None],
    "salary": [55000, 72000, 83000, 48000, 51000, None, 67000, 45000, 53000, 46000, 85000, 75000, 49000, 68000, 54000, 62000, 69000, 71000, 54000, 56000, -454545, None],
    "total_rooms": [6, 4, 8, 3, 0, 7, 5, 6, 4, 3, 9, 7, 4, 5, 6, 6, 7, 6, 5, 6, 4, 7]
}
df = pd.DataFrame(house_data)

csv_file_path = "/content/house_data.csv"
df.to_csv(csv_file_path, index=False)
df = pd.read_csv(csv_file_path)

valid_ages = df['age'][df['age'] > 0]
mean_age = valid_ages.mean()

df['age'] = df['age'].apply(lambda x: mean_age if pd.isna(x) or x < 0 else x)

print(df)

mode_owns = df['owns'].mode()[0]
df['owns'] = df['owns'].apply(lambda x: mode_owns if x is None else x)
print(df)

valid_salarys = df['salary'][df['salary'] > 0]
mean_salary = valid_salarys.mean()

df['salary'] = df['salary'].apply(lambda x: mean_salary if pd.isna(x) or x < 0 else x)

print(df)
