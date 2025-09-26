# Elevate-Labs-Task-2

# 🚢 Titanic Dataset - Exploratory Data Analysis (EDA)

This project explores the Titanic dataset to uncover patterns, clean missing data, and visualize key features using statistical techniques.

## 📌 Objective
Perform Exploratory Data Analysis (EDA) on the **Titanic-Dataset.csv** to:
- Clean and preprocess the data.
- Understand the distribution of features.
- Analyze relationships and survival patterns.

## 🛠️ Tools & Libraries Used
- **Pandas**: Data manipulation
- **Matplotlib** & **Seaborn**: Data visualization
- **Scipy**: Statistical operations (Z-score)
- **Scikit-learn**: Data preprocessing (encoding)

## 🚀 How to Run

1. Install required libraries:
   ```bash
   pip install pandas matplotlib seaborn scikit-learn
   ```

2. Launch the notebook:
   ```bash
   jupyter notebook "Task 2.ipynb"
   ```

## 📁 Dataset
**File**: `Titanic-Dataset.csv`  
Make sure the dataset is present in the same directory as the notebook.

---

## 🔍 EDA Steps

### 1. **Load and Inspect Data**
- Loaded the dataset using `pandas`
- Displayed rows with missing values

### 2. **Handle Missing Values**
- **Age**: Imputed with median  
- **Embarked**: Imputed with mode  
- **Cabin**: Imputed with `"Unknown"`

### 3. **Summary Statistics**
- Displayed **mean**, **median**, **mode**, and other stats
- Selected key columns for focused analysis:
  - `Survived`, `Age`, `SibSp`, `Sex`, `Embarked`, `Fare`, `Parch`

### 4. **Visualizations**
- **Histograms** and **Boxplots** for numeric features: `Age`, `Fare`, `SibSp`, `Parch`
- **Correlation heatmap** of selected features
- **Pairplot** of variables colored by survival status
- **Barplot** and **Countplot**:
  - Survival rate by passenger class
  - Survival count by sex

### 5. **Encoding**
- Converted `Sex` column to numerical format: male → 0, female → 1

### 6. **Outlier Detection**
- Used **Z-score** method to detect outliers in the `Fare` column

### 7. **Group-wise Survival Analysis**
- Survival rate by:
  - Sex
  - Passenger class and sex

---

## ✅ Outcome
This EDA uncovers important relationships and data characteristics, setting a solid foundation for further machine learning modeling or hypothesis testing.
