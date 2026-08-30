# Health Condition Classification — Experiment 1

## Overview

This experiment establishes a **baseline machine learning pipeline** for multiclass health-condition classification.

The experiment uses:

* Missing-value handling
* Correlation analysis
* Z-score normalization
* Probability encoding for categorical features
* Label encoding
* Logistic Regression
* Accuracy
* Precision, Recall, F1-score
* Confusion Matrix
* Multiclass ROC-AUC using One-vs-Rest (OvR)

The purpose of this experiment is to create a **simple baseline model** before moving to more advanced techniques such as class balancing, feature selection, hyperparameter optimization, and nonlinear models.

---

## Project Structure

```text
health_condition/
│
├── dataset/
│   ├── train.csv
│   ├── test.csv
│   └── sample_submission.csv
│
└── 1-experement/
    │
    ├── main.py
    │
    └── model/
        ├── model.py
        └── preprocessing.py
```

---

# Pipeline

The complete pipeline follows these steps:

```text
Raw Dataset
     │
     ▼
Load Data
     │
     ▼
Handle Missing Values
     │
     ▼
Correlation Analysis
     │
     ▼
Z-Score Normalization
     │
     ▼
Categorical Feature Encoding
     │
     ▼
Label Encoding
     │
     ▼
Logistic Regression
     │
     ▼
Model Evaluation
     │
     ├── Accuracy
     ├── Classification Report
     ├── Confusion Matrix
     └── Multiclass ROC-AUC
```

---

# 1. Data Loading

The `load_data()` function loads the training and test datasets from CSV files.

The `id` column is removed from the input features, while `health_condition` is used as the target variable for the training data.

```python
X_train = train_df.drop(columns=['id', 'health_condition']).values
Y_train = train_df['health_condition'].values
```

For the test set, the `id` column is removed and the labels are obtained from the sample submission file.

---

# 2. Handling Missing Values

Numeric and categorical features are processed separately.

The current experiment uses **mean imputation** for numerical features.

The mean is calculated from the training data and then applied to both training and test data:

```python
mean_values = np.nanmean(x_numeric_train, axis=0)
```

Missing categorical values are filled using the **mode** of each categorical feature.

### Current configuration

```python
handle_missing_values(
    X_train,
    X_test,
    strategy='mean'
)
```

The implementation also supports:

```python
strategy='median'
```

---

# 3. Correlation Analysis

Correlation analysis is performed on the numerical features.

The experiment uses a threshold of:

```text
0.8
```

Features with an absolute correlation greater than this threshold are reported as highly correlated.

```python
corr_matrix = check_correlation(
    X_train,
    threshold=0.8
)
```

This step is currently **analytical**: the pipeline reports highly correlated features but does not automatically remove them.

---

# 4. Z-Score Normalization

Numerical features are normalized using Z-score standardization.

The mean and standard deviation are calculated from the training data:

```python
mean = np.mean(x_numeric_train, axis=0)
std = np.std(x_numeric_train, axis=0)
```

The same training statistics are then applied to the test data.

Conceptually:

```text
                    x - μ
Z =                 ─────
                      σ
```

This is particularly useful for Logistic Regression because the model is sensitive to the scale of input features.

---

# 5. Categorical Feature Encoding

Categorical features are converted into numerical values using **probability encoding**.

For each categorical feature, the frequency/probability of each category is calculated from the training data.

For example:

```text
Category      Probability
-------------------------
A                 0.60
B                 0.30
C                 0.10
```

The category is then replaced by its corresponding probability.

The mappings are generated from the training data and applied to both training and test features.

Unknown categories in the test data are assigned:

```text
0
```

---

# 6. Label Encoding

The target labels are converted into numerical classes using `LabelEncoder`.

For the current multiclass problem, the target contains three classes:

```text
Class 0
Class 1
Class 2
```

---

# 7. Logistic Regression

The baseline model is Logistic Regression.

Current configuration:

```python
LogisticRegression(
    penalty='l2',
    C=1.0,
    solver='liblinear',
    random_state=42
)
```

The model is trained using the processed training data.

### Parameters

| Parameter      |       Value | Description                     |
| -------------- | ----------: | ------------------------------- |
| `penalty`      |        `l2` | L2 regularization               |
| `C`            |       `1.0` | Inverse regularization strength |
| `solver`       | `liblinear` | Optimization algorithm          |
| `random_state` |        `42` | Reproducibility                 |

The implementation also supports L1 regularization:

```python
penalty='l1'
```

When L1 regularization is used, the number of non-zero coefficients is reported.

---

# 8. Model Evaluation

The model produces both class predictions and class probabilities:

```python
Y_pred = model.predict(X)
Y_pred_proba = model.predict_proba(X)
```

The following metrics are calculated:

### Accuracy

Measures the percentage of correctly classified samples.

### Classification Report

Reports:

* Precision
* Recall
* F1-score
* Support

for each class.

### Confusion Matrix

Shows how samples from each actual class are classified by the model.

### Multiclass ROC-AUC

Because this is a three-class classification problem, ROC-AUC is calculated using:

```python
multi_class='ovr'
average='macro'
```

This evaluates each class against all other classes and then calculates the macro-average AUC.

---

# 9. Baseline Results

The first experiment produced the following results:

```text
Accuracy: 0.8716

AUC-ROC (OvR): 0.9054
```

### Classification Report

```text
              precision    recall  f1-score   support

           0       0.88      0.98      0.93    592561
           1       0.75      0.10      0.17     39803
           2       0.64      0.26      0.36     57724

    accuracy                           0.87    690088
   macro avg       0.76      0.44      0.49    690088
weighted avg       0.85      0.87      0.84    690088
```

### Confusion Matrix

```text
[[582949   1261   8351]
 [ 35975   3791     37]
 [ 42952     15  14757]]
```

---

# 10. Initial Analysis

Although the baseline achieves:

```text
Accuracy = 87.16%
AUC      = 90.54%
```

the classification report shows a significant difference between the three classes.

Class `0` has very high recall:

```text
Recall = 0.98
```

while Class `1` has:

```text
Recall = 0.10
```

and Class `2` has:

```text
Recall = 0.26
```

This indicates that the model performs well on the majority class but struggles to identify the minority classes.

The macro F1-score is:

```text
0.49
```

which provides a more informative view of the model's performance across the three classes than accuracy alone.

---

# 11. Limitations of Experiment 1

This experiment is intentionally a **baseline**.

Several areas remain to be investigated:

* Class imbalance
* Undersampling of the majority class
* Class weighting
* L1 vs L2 regularization
* Different values of `C`
* Feature selection
* More detailed threshold analysis
* Nonlinear machine learning models
* Cross-validation
* Improved evaluation methodology

The next experiments will build on this baseline and determine which techniques improve minority-class performance without unnecessarily sacrificing overall performance.

---

# 12. Running the Experiment

From the `1-experement` directory:

```bash
python main.py
```

The main pipeline is implemented in `main.py`, which calls the preprocessing functions, trains Logistic Regression, evaluates the model, and stores the AUC result.

---

# Experiment Goal

The goal of Experiment 1 is **not to produce the final model**.

Instead, it establishes a reproducible baseline against which future experiments can be compared.

Future experiments should report at least:

```text
Accuracy
Macro F1
Class 0 F1
Class 1 F1
Class 2 F1
Recall per class
AUC-ROC
Confusion Matrix
```

This allows each modification to the pipeline to be evaluated objectively.

