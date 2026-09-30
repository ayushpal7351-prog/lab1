# MCA201P – LAB SHEET 02
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import (
    MinMaxScaler,
    StandardScaler,
    RobustScaler,
    MaxAbsScaler,
    LabelEncoder,
    OneHotEncoder
)

# Load dataset
df = pd.read_csv("data.csv")

# Keep original copy
original_df = df.copy()
```

---

# 1. Load Dataset and Identify Missing Values

```python
import pandas as pd

df = pd.read_csv("data.csv")

missing_values = df.isnull().sum()

print(missing_values)
```

---

# 2. Display Percentage of Missing Values

```python
missing_percentage = (df.isnull().sum() / len(df)) * 100

print(missing_percentage)
```

---

# 3. Remove Rows Containing Missing Values

```python
df_cleaned = df.copy()

df_cleaned = df_cleaned.dropna()

print(df_cleaned)
```

---

# 4. Remove Columns Having More Than 50% Missing Values

```python
df_cleaned = df.copy()

threshold = 0.50

missing_ratio = df_cleaned.isnull().mean()

columns_to_remove = missing_ratio[
    missing_ratio > threshold
].index

df_cleaned = df_cleaned.drop(columns=columns_to_remove)

print(df_cleaned)
```

---

# 5. Replace Missing Numerical Values Using Mean

```python
df_mean = df.copy()

numerical_columns = df_mean.select_dtypes(
    include=np.number
).columns

for column in numerical_columns:
    df_mean[column] = df_mean[column].fillna(
        df_mean[column].mean()
    )

print(df_mean)
```

---

# 6. Replace Missing Numerical Values Using Median

```python
df_median = df.copy()

numerical_columns = df_median.select_dtypes(
    include=np.number
).columns

for column in numerical_columns:
    df_median[column] = df_median[column].fillna(
        df_median[column].median()
    )

print(df_median)
```

---

# 7. Replace Missing Categorical Values Using Mode

```python
df_mode = df.copy()

categorical_columns = df_mode.select_dtypes(
    include="object"
).columns

for column in categorical_columns:
    if not df_mode[column].mode().empty:
        df_mode[column] = df_mode[column].fillna(
            df_mode[column].mode()[0]
        )

print(df_mode)
```

---

# 8. Fill Missing Values Using Forward Fill

```python
df_forward = df.copy()

df_forward = df_forward.ffill()

print(df_forward)
```

---

# 9. Fill Missing Values Using Backward Fill

```python
df_backward = df.copy()

df_backward = df_backward.bfill()

print(df_backward)
```

---

# 10. Compare Dataset Before and After Handling Missing Values

```python
before_missing = df.copy()

after_missing = df.copy()

# Fill numerical columns with median
numerical_columns = after_missing.select_dtypes(
    include=np.number
).columns

for column in numerical_columns:
    after_missing[column] = after_missing[column].fillna(
        after_missing[column].median()
    )

# Fill categorical columns with mode
categorical_columns = after_missing.select_dtypes(
    include="object"
).columns

for column in categorical_columns:
    if not after_missing[column].mode().empty:
        after_missing[column] = after_missing[column].fillna(
            after_missing[column].mode()[0]
        )

print("Missing values before:")
print(before_missing.isnull().sum())

print("Missing values after:")
print(after_missing.isnull().sum())
```

---

# 11. Detect Outliers Using IQR Method

```python
df_iqr = df.copy()

numerical_columns = df_iqr.select_dtypes(
    include=np.number
).columns

for column in numerical_columns:

    Q1 = df_iqr[column].quantile(0.25)
    Q3 = df_iqr[column].quantile(0.75)

    IQR = Q3 - Q1

    lower_limit = Q1 - 1.5 * IQR
    upper_limit = Q3 + 1.5 * IQR

    outliers = df_iqr[
        (df_iqr[column] < lower_limit) |
        (df_iqr[column] > upper_limit)
    ]

    print("Column:", column)
    print("Number of outliers:", len(outliers))
```

---

# 12. Detect Outliers Using Z-Score Method

```python
from scipy.stats import zscore

df_zscore = df.copy()

numerical_columns = df_zscore.select_dtypes(
    include=np.number
).columns

for column in numerical_columns:

    valid_values = df_zscore[column].dropna()

    z_scores = np.abs(zscore(valid_values))

    outliers = valid_values[z_scores > 3]

    print("Column:", column)
    print("Number of outliers:", len(outliers))
```

---

# 13. Visualize Outliers Using Box Plot

```python
df_box = df.copy()

numerical_columns = df_box.select_dtypes(
    include=np.number
).columns

for column in numerical_columns:

    plt.figure(figsize=(8, 5))

    sns.boxplot(x=df_box[column])

    plt.title("Box Plot - " + column)
    plt.xlabel(column)

    plt.show()
```

---

# 14. Visualize Outliers Using Scatter Plot

```python
df_scatter = df.copy()

numerical_columns = df_scatter.select_dtypes(
    include=np.number
).columns

if len(numerical_columns) >= 2:

    x_column = numerical_columns[0]
    y_column = numerical_columns[1]

    plt.figure(figsize=(8, 5))

    plt.scatter(
        df_scatter[x_column],
        df_scatter[y_column]
    )

    plt.xlabel(x_column)
    plt.ylabel(y_column)
    plt.title("Scatter Plot for Outlier Detection")

    plt.show()
```

---

# 15. Remove Outliers Using IQR Method

```python
df_without_outliers = df.copy()

numerical_columns = df_without_outliers.select_dtypes(
    include=np.number
).columns

for column in numerical_columns:

    Q1 = df_without_outliers[column].quantile(0.25)
    Q3 = df_without_outliers[column].quantile(0.75)

    IQR = Q3 - Q1

    lower_limit = Q1 - 1.5 * IQR
    upper_limit = Q3 + 1.5 * IQR

    df_without_outliers = df_without_outliers[
        (df_without_outliers[column] >= lower_limit) &
        (df_without_outliers[column] <= upper_limit)
    ]

print(df_without_outliers)
```

---

# 16. Replace Outliers with Median Value

```python
df_median_outliers = df.copy()

numerical_columns = df_median_outliers.select_dtypes(
    include=np.number
).columns

for column in numerical_columns:

    Q1 = df_median_outliers[column].quantile(0.25)
    Q3 = df_median_outliers[column].quantile(0.75)

    IQR = Q3 - Q1

    lower_limit = Q1 - 1.5 * IQR
    upper_limit = Q3 + 1.5 * IQR

    median_value = df_median_outliers[column].median()

    outlier_condition = (
        (df_median_outliers[column] < lower_limit) |
        (df_median_outliers[column] > upper_limit)
    )

    df_median_outliers.loc[
        outlier_condition,
        column
    ] = median_value

print(df_median_outliers)
```

---

# 17. Cap Outliers Using Percentile-Based Capping

```python
df_capped = df.copy()

numerical_columns = df_capped.select_dtypes(
    include=np.number
).columns

for column in numerical_columns:

    lower_limit = df_capped[column].quantile(0.01)
    upper_limit = df_capped[column].quantile(0.99)

    df_capped[column] = df_capped[column].clip(
        lower_limit,
        upper_limit
    )

print(df_capped)
```

---

# 18. Compare Dataset Before and After Outlier Treatment

```python
before_outlier_treatment = df.copy()

after_outlier_treatment = df.copy()

numerical_columns = after_outlier_treatment.select_dtypes(
    include=np.number
).columns

for column in numerical_columns:

    Q1 = after_outlier_treatment[column].quantile(0.25)
    Q3 = after_outlier_treatment[column].quantile(0.75)

    IQR = Q3 - Q1

    lower_limit = Q1 - 1.5 * IQR
    upper_limit = Q3 + 1.5 * IQR

    after_outlier_treatment[column] = (
        after_outlier_treatment[column]
        .clip(lower_limit, upper_limit)
    )

print("Before outlier treatment:")
print(before_outlier_treatment.describe())

print("After outlier treatment:")
print(after_outlier_treatment.describe())
```

---

# 19. Apply Min-Max Normalization

```python
df_minmax = df.copy()

numerical_columns = df_minmax.select_dtypes(
    include=np.number
).columns

scaler = MinMaxScaler()

df_minmax[numerical_columns] = scaler.fit_transform(
    df_minmax[numerical_columns]
)

print(df_minmax)
```

---

# 20. Apply Standardization (Z-Score Scaling)

```python
df_standard = df.copy()

numerical_columns = df_standard.select_dtypes(
    include=np.number
).columns

scaler = StandardScaler()

df_standard[numerical_columns] = scaler.fit_transform(
    df_standard[numerical_columns]
)

print(df_standard)
```

---

# 21. Apply Robust Scaling

```python
df_robust = df.copy()

numerical_columns = df_robust.select_dtypes(
    include=np.number
).columns

scaler = RobustScaler()

df_robust[numerical_columns] = scaler.fit_transform(
    df_robust[numerical_columns]
)

print(df_robust)
```

---

# 22. Apply Max Absolute Scaling

```python
df_maxabs = df.copy()

numerical_columns = df_maxabs.select_dtypes(
    include=np.number
).columns

scaler = MaxAbsScaler()

df_maxabs[numerical_columns] = scaler.fit_transform(
    df_maxabs[numerical_columns]
)

print(df_maxabs)
```

---

# 23. Compare Original and Normalized Datasets

```python
original_data = df.copy()

normalized_data = df.copy()

numerical_columns = normalized_data.select_dtypes(
    include=np.number
).columns

scaler = MinMaxScaler()

normalized_data[numerical_columns] = scaler.fit_transform(
    normalized_data[numerical_columns]
)

print("Original Dataset:")
print(original_data)

print("Normalized Dataset:")
print(normalized_data)
```

---

# 24. Visualize Effect of Normalization Using Histograms

```python
df_hist = df.copy()

numerical_columns = df_hist.select_dtypes(
    include=np.number
).columns

scaler = MinMaxScaler()

normalized_values = scaler.fit_transform(
    df_hist[numerical_columns]
)

normalized_df = pd.DataFrame(
    normalized_values,
    columns=numerical_columns
)

for column in numerical_columns:

    plt.figure(figsize=(8, 5))

    plt.hist(
        df_hist[column].dropna(),
        alpha=0.5,
        label="Original"
    )

    plt.hist(
        normalized_df[column].dropna(),
        alpha=0.5,
        label="Normalized"
    )

    plt.title("Original vs Normalized - " + column)
    plt.xlabel(column)
    plt.ylabel("Frequency")
    plt.legend()

    plt.show()
```

---

# 25. Visualize Effect of Scaling Using Box Plots

```python
df_scaling = df.copy()

numerical_columns = df_scaling.select_dtypes(
    include=np.number
).columns

scaler = StandardScaler()

scaled_values = scaler.fit_transform(
    df_scaling[numerical_columns]
)

scaled_df = pd.DataFrame(
    scaled_values,
    columns=numerical_columns
)

for column in numerical_columns:

    plt.figure(figsize=(8, 5))

    plt.boxplot(
        scaled_df[column].dropna()
    )

    plt.title("Scaled Data - " + column)
    plt.ylabel("Scaled Value")

    plt.show()
```

---

# 26. Compare Different Scaling Techniques

```python
numerical_columns = df.select_dtypes(
    include=np.number
).columns

data = df[numerical_columns].fillna(
    df[numerical_columns].median()
)

minmax = MinMaxScaler().fit_transform(data)
standard = StandardScaler().fit_transform(data)
robust = RobustScaler().fit_transform(data)
maxabs = MaxAbsScaler().fit_transform(data)

comparison = pd.DataFrame({
    "MinMax": minmax[:, 0],
    "Standard": standard[:, 0],
    "Robust": robust[:, 0],
    "MaxAbs": maxabs[:, 0]
})

print(comparison)
```

---

# 27. Label Encoding

```python
df_label = df.copy()

label_encoder = LabelEncoder()

categorical_columns = df_label.select_dtypes(
    include="object"
).columns

for column in categorical_columns:

    df_label[column] = label_encoder.fit_transform(
        df_label[column].astype(str)
    )

print(df_label)
```

---

# 28. One-Hot Encoding

```python
df_onehot = df.copy()

categorical_columns = df_onehot.select_dtypes(
    include="object"
).columns

df_onehot = pd.get_dummies(
    df_onehot,
    columns=categorical_columns,
    dtype=int
)

print(df_onehot)
```

---

# 29. Binary Encoding on Categorical Data

```python
# Install first if required:
# pip install category_encoders

import category_encoders as ce

df_binary = df.copy()

categorical_columns = df_binary.select_dtypes(
    include="object"
).columns

binary_encoder = ce.BinaryEncoder(
    cols=list(categorical_columns)
)

df_binary = binary_encoder.fit_transform(
    df_binary
)

print(df_binary)
```

---

# 30. Create a New Feature by Combining Two Columns

```python
df_combined = df.copy()

# Example: combine two columns
# Change column names according to your dataset

df_combined["Combined_Feature"] = (
    df_combined["Column1"].astype(str)
    + "_"
    + df_combined["Column2"].astype(str)
)

print(df_combined)
```

---

# 31. Extract Year, Month and Day from a Date Column

```python
df_date = df.copy()

df_date["Date"] = pd.to_datetime(
    df_date["Date"],
    errors="coerce"
)

df_date["Year"] = df_date["Date"].dt.year
df_date["Month"] = df_date["Date"].dt.month
df_date["Day"] = df_date["Date"].dt.day

print(df_date)
```

---

# 32. Create a New Feature Using Mathematical Transformations

```python
df_math = df.copy()

numerical_columns = df_math.select_dtypes(
    include=np.number
).columns

column_name = numerical_columns[0]

df_math["Squared_Feature"] = (
    df_math[column_name] ** 2
)

print(df_math)
```

---

# 33. Apply Log Transformation to Skewed Data

```python
df_log = df.copy()

numerical_columns = df_log.select_dtypes(
    include=np.number
).columns

column_name = numerical_columns[0]

df_log["Log_Transformed"] = np.log1p(
    df_log[column_name]
)

print(df_log)
```

---

# 34. Feature Selection Using Correlation Analysis

```python
df_correlation = df.copy()

numerical_data = df_correlation.select_dtypes(
    include=np.number
)

correlation_matrix = numerical_data.corr()

plt.figure(figsize=(10, 7))

sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap="coolwarm"
)

plt.title("Correlation Matrix")

plt.show()

# Select highly correlated features
correlation_with_target = (
    correlation_matrix.iloc[:, -1]
    .abs()
    .sort_values(ascending=False)
)

print(correlation_with_target)
```

---

# 35. Create Final Preprocessed Dataset

```python
final_df = df.copy()

# Separate numerical and categorical columns
numerical_columns = final_df.select_dtypes(
    include=np.number
).columns

categorical_columns = final_df.select_dtypes(
    include="object"
).columns

# Handle missing numerical values
for column in numerical_columns:
    final_df[column] = final_df[column].fillna(
        final_df[column].median()
    )

# Handle missing categorical values
for column in categorical_columns:
    if not final_df[column].mode().empty:
        final_df[column] = final_df[column].fillna(
            final_df[column].mode()[0]
        )

# Remove duplicate records
final_df = final_df.drop_duplicates()

# Apply Min-Max scaling to numerical columns
scaler = MinMaxScaler()

final_df[numerical_columns] = scaler.fit_transform(
    final_df[numerical_columns]
)

# Encode categorical columns
final_df = pd.get_dummies(
    final_df,
    columns=categorical_columns,
    dtype=int
)

# Save final preprocessed dataset
final_df.to_csv(
    "final_preprocessed_dataset.csv",
    index=False
)

print("Final preprocessing completed.")
```

---

# Conclusion

The above 35 programs cover the complete data preprocessing practical specified in the lab sheet. The experiments include missing-value handling, outlier detection and treatment, normalization, scaling, categorical encoding, feature creation, date extraction, mathematical transformation, logarithmic transformation, feature selection, and preparation of a final dataset for Machine Learning.
