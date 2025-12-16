#====================================
#-------- PREPROCESSING DATA --------
#====================================

import pandas as pd
from sklearn.model_selection import train_test_split

# Load dataset
file_path = "data/lung_cancer.csv"
df = pd.read_csv(file_path, sep=';')

# Hapus duplikat
duplikat = df.duplicated().sum()
df = df.drop_duplicates()

# Normalisasi string
df = df.applymap(lambda x: x.upper() if isinstance(x, str) else x)

# Rename kolom
df.rename(columns=lambda x: x.strip().upper().replace(" ", "_"), inplace=True)

# Konversi nilai 1 dan 2
def convert_1_2(val):
    if val == 1:
        return 0
    elif val == 2:
        return 1
    else:
        return val

for col in df.columns:
    if df[col].dtype in ['int64', 'float64']:
        df[col] = df[col].apply(convert_1_2)

# Labeling gender
df['GENDER'] = df['GENDER'].replace({'M': 1, 'F': 0})

# Konversi yes no menjadi 0 1
df.replace({'YES': 1, 'NO': 0}, inplace=True)

X = df.drop(columns=['LUNG_CANCER'])
y = df['LUNG_CANCER']

# Save ke file baru
df.to_csv("data/cleaned_lung_cancer.csv", index=False)