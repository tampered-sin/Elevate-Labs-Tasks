# Elevate-Labs-Task-1
# 🛠️ Titanic Dataset Preprocessing – README

This project involves cleaning and preparing the Titanic dataset for machine learning by handling missing values, encoding categorical data, scaling numerical features, and removing outliers.

---

## 📦 Dataset

- Source: `Titanic-Dataset.csv`
- Common features used: `Age`, `Fare`, `Sex`, `Embarked`, `SibSp`, `Parch`, etc.

---

## ✅ Preprocessing Steps

### 1. **Load the Dataset**
```python
import pandas as pd
df = pd.read_csv("Titanic-Dataset.csv")
```

---

### 2. **Identify Missing Values**
```python
df.isnull().sum()
```

---

### 3. **Handle Missing Values**
- **Age**: Imputed with **median**
- **Embarked**: Imputed with **mode**
- **Cabin**: Imputed with `"Unknown"`

```python
df_imputed["Age"] = df_imputed["Age"].fillna(df_imputed["Age"].median())
df_imputed["Embarked"] = df_imputed["Embarked"].fillna(df_imputed["Embarked"].mode()[0])
df_imputed["Cabin"] = df_imputed["Cabin"].fillna("Unknown")
```

---

### 4. **Round the Age Column**
```python
df_imputed["Age"] = df_imputed["Age"].round(0).astype("Int64")
```

---

### 5. **Encode Categorical Variables**
- **Sex**: Label encoded (`male` → 0, `female` → 1)
- **Embarked**: One-hot encoded (`Embarked_C`,`Embarked_Q`, `Embarked_S`)
```python
df_encoded["Sex"] = df_encoded["Sex"].map({"male": 0, "female": 1})
df_encoded = pd.get_dummies(df_encoded, columns=["Embarked"], drop_first=False)
```

---

### 6. **Scale Numerical Features**
**Standardized** (`Age`, `Fare`, `SibSp`, `Parch`) using `StandardScaler`

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
num_cols = ["Age", "Fare", "SibSp", "Parch"]
df_standardized[num_cols] = scaler.fit_transform(df_standardized[num_cols])
```

---

### 7. **Visualize & Remove Outliers**
- Used **boxplots** and **IQR method** to detect and remove outliers in numerical features

```python
# Remove outliers using IQR
def remove_outliers(df, col):
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    IQR = q3 - q1
    return df[(df[col] >= q1 - 1.5 * IQR) & (df[col] <= q3 + 1.5 * IQR)]

for col in num_cols:
    df_outlier = remove_outliers(df_outlier, col)
```

---


## ✅ Final Output

- Cleaned, encoded, scaled, and outlier-free dataset
- Ready for machine learning models
