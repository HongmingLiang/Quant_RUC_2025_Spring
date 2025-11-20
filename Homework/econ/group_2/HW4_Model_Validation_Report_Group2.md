
Authors: Group 2 (白瑞睿, 陈君昊, 郑瑜)

Target Project: Team 3 Midterm Codes (`Midterm_codes_Team3.ipynb`)  

## **1. Replicability (Run Codes)**

The results are replicable, but the execution contains warnings that suggest data integrity risks early in the pipeline.

**Specific Issue (DtypeWarning):** Upon execution, the following warning is generated:  `DtypeWarning: Columns (3,32,34,43,46,49,51) have mixed types. Specify dtype option on import or set low\_memory=False.`

This indicates that the dataset contains columns where integers/floats are mixed with strings. While the code does not crash immediately, this causes Pandas to assign the object type to these columns. This consumes significantly more memory.

## **2. Data Processing Check**

Group 3 treated the two tasks as distinct problems with separate pipelines, effectively addressing different features and feature distributions of the data.

The team handles non-standard missing values effectively.
- They identify categorical columns and fill missing values with "Unknown". For numerical features, they use median imputation.
- They use Regex parsing (e.g., `parse_build_year`, `parse_property_fee`) which implicitly handles messy string data (like "Unknown" or "-") by returning `np.nan`, which is then filled later. This is a robust approach.

## **3. Sample Choice & Data Leakage**

`train_test_split` is used with `test_size=0.2` and a fixed `random_state=111`. The split is performed correctly.

**Data Leakage Check:** We rigorously checked if information from the validation/test set leaked into the training process. The team followed best practices:
1. **Imputation (Cell 10):**
	- Method: They calculate `median_value = X_train[col].median()` and then apply this saved value to `X_val` and `X_test`.
	- No Leakage.
2. **Target Encoding (Cell 28-30):**
	- Method: They calculate mean prices (`groupby('城市')['Price']`) using **only** `train_data_for_encoding`.
	- Application: They map these values to `X_val` and `X_test`. Crucially, for the validation/test sets, they handle unseen categories by filling them with the `global_mean` calculated from the train set.
	- No Leakage.
3. **Scaling (Cell 42):**
	- Method: They initialize `StandardScaler`, call `scaler.fit(X_train)`, and then `scaler.transform(X_val)`.
	- No Leakage.

## **4. Suggestions for Group 3**

### **Suggestion 1: Address DtypeWarning Explicitly**

**Current Issue:** In Cell 2 and Cell 49, the `DtypeWarning` implies inefficient memory usage and potential type errors later. 

**Recommendation:** Specify `dtype` or `low_memory=False` during `read_csv`.
```
# Example fix  
df_train_price = pd.read_csv('ruc_Class25Q2_train_price.csv', low_memory=False)  
# OR explicitly specify types for the flagged columns
```

### **Suggestion 2: Optimize Feature Selection**

**Current Issue:** In Cell 44 (LassoCV), the model selects 167 out of 173 features. While Lasso performs selection, retaining 96% of features (including many One-Hot Encoded variables and interaction terms) suggests the penalty might be too loose or the feature space is noisy.

**Recommendation:** Consider removing features with extremely low variance or high correlation before modeling to reduce model complexity and training time (which took ~30 minutes for LassoCV in Cell 44).

### **Suggestion 3: Code Modularity**

**Current Issue:** The notebook is very long, with repetitive code blocks for Price and Rent processing (e.g., Cell 42 and Cell 74 are nearly identical logic but for different datasets).

**Recommendation:** Define a set of functions that takes the dataframe as an input. This reduces code duplication and the risk of copy-paste errors.

