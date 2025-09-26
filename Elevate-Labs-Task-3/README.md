# Elevate-Labs-Task-3
### Author: Tenzin Kunga
### Date: 2025-05-29
### Objective: Implement and understand simple & multiple linear regression.
### Tools: Scikit-learn, Pandas, Matplotlib
---
### Import necessary libraries
```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

```
---
### Load the dataset
```py
data = pd.read_csv("Housing.csv")
```

