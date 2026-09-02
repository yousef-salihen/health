# Experiment 2 — Balanced ADALINE Classification

This experiment improves the baseline classification pipeline by addressing **class imbalance**, removing outliers, applying feature normalization, and training a multiclass **ADALINE** model using a One-vs-Rest strategy.

The main objective was not only to maximize accuracy, but to improve the model's ability to correctly identify **all three classes**.

---

## Experiment Overview

The original training dataset contained a significant class imbalance, with the majority class representing most of the available samples.

Instead of training directly on the imbalanced dataset, this experiment applies the following pipeline:

```text
Raw Training Data
       ↓
Categorical Feature Encoding
       ↓
Label Encoding
       ↓
Class Balancing
       ↓
Outlier Detection & Removal
       ↓
Z-Score Normalization
       ↓
ADALINE (One-vs-Rest)
       ↓
Multiclass Prediction
       ↓
Evaluation
```

---

## Dataset

The original dataset contained:

```text
Training samples: 690,088
Features:         13
```

The testing dataset contained:

```text
Testing samples: 295,753
Features:        13
```

After balancing the training data:

```text
Balanced training samples: 119,409
Features:                   13
```

After detecting and removing outliers:

```text
Training samples after outlier removal: 106,888
Features:                                13
```

---

## Preprocessing

### 1. Categorical Feature Encoding

Categorical features were converted into numerical representations so that they could be processed by the ADALINE model.

```text
Categorical Features
        ↓
Numerical Representation
```

---

### 2. Label Encoding

The target labels were encoded into three classes:

```text
Classes: [0, 1, 2]
```

This makes the classification problem suitable for the multiclass ADALINE implementation.

---

### 3. Class Balancing

The original dataset was highly imbalanced.

To prevent the model from being biased toward the majority class, the training data was balanced before training.

```text
Original Training Data
        ↓
Class Balancing
        ↓
119,409 samples
```

This step was particularly important because accuracy alone can be misleading when the target classes are not equally represented.

---

### 4. Outlier Detection

After balancing the dataset, outlier detection was applied to remove extreme observations that could negatively affect the linear ADALINE decision boundaries.

```text
119,409 samples
        ↓
Outlier Detection
        ↓
106,888 samples
```

---

### 5. Z-Score Normalization

The numerical features were normalized using Z-score normalization.

The transformation is:

```text
z = (x - μ) / σ
```

where:

* `x` = original feature value
* `μ` = feature mean
* `σ` = feature standard deviation

Normalization is especially important for ADALINE because its optimization process is sensitive to the scale of the input features.

---

# ADALINE Model

ADALINE (Adaptive Linear Neuron) is a linear neural model trained using the **Least Mean Squares (LMS)** learning rule.

For this experiment, a separate ADALINE classifier was trained for each class using a **One-vs-Rest (OvR)** approach.

```text
                 ┌── ADALINE → Class 0
Input Features ──┼── ADALINE → Class 1
                 └── ADALINE → Class 2
```

During prediction, the outputs of the three classifiers are compared and the class with the strongest response is selected.

Training output:

```text
Classes: [0 1 2]

Training ADALINE for class: 0
Training ADALINE for class: 1
Training ADALINE for class: 2

ADALINE training completed.
```

---

# Results

## Accuracy

```text
Accuracy: 0.7548
```

The model achieved:

**75.48% accuracy**

---

## AUC-ROC

Using a One-vs-Rest evaluation:

```text
AUC-ROC (OvR): 0.8795
```

The model achieved an overall **AUC-ROC of 87.95%**.

---

## Classification Report

```text
              precision    recall  f1-score   support

           0       0.80      0.44      0.57     35793
           1       0.73      0.94      0.82     36211
           2       0.76      0.89      0.82     34884

    accuracy                           0.75    106888
   macro avg       0.76      0.76      0.74    106888
weighted avg       0.76      0.75      0.74    106888
```

### Key observations

The model achieved relatively balanced performance across the three classes.

In particular:

```text
Class 1 Recall = 0.94
Class 2 Recall = 0.89
```

This means the model successfully identifies most samples belonging to the minority classes.

The macro F1-score reached:

```text
Macro F1 = 0.74
```

which provides a better representation of the model's performance across all classes than accuracy alone.

---

# Confusion Matrix

```text
[[15703 10916  9174]
 [ 1578 33928   705]
 [ 2330  1509 31045]]
```

The diagonal values represent correctly classified samples.

```text
Class 0 → 15,703 correctly classified
Class 1 → 33,928 correctly classified
Class 2 → 31,045 correctly classified
```

The model performs particularly well on Classes 1 and 2.

---

# Comparison with Baseline

The first experiment achieved:

```text
Accuracy:      0.8716
AUC-ROC:       0.9054
Macro Recall:  0.44
Macro F1:      0.49
```

The second experiment achieved:

```text
Accuracy:      0.7548
AUC-ROC:       0.8795
Macro Recall:  0.76
Macro F1:      0.74
```

| Metric          |   Baseline | Experiment 2 |
| --------------- | ---------: | -----------: |
| Accuracy        | **87.16%** |       75.48% |
| AUC-ROC         | **90.54%** |       87.95% |
| Macro Precision |        76% |      **76%** |
| Macro Recall    |        44% |      **76%** |
| Macro F1        |        49% |      **74%** |

Although the second experiment has lower overall accuracy, it provides a much more balanced classification performance.

The baseline model achieved high accuracy largely because of its strong performance on the majority class.

For example:

```text
Baseline:

Class 0 Recall = 0.98
Class 1 Recall = 0.10
Class 2 Recall = 0.26
```

In contrast:

```text
Experiment 2:

Class 0 Recall = 0.44
Class 1 Recall = 0.94
Class 2 Recall = 0.89
```

Therefore, the second experiment significantly improves the detection of the underrepresented classes.

---

# Experiment Conclusion

The second experiment demonstrates the impact of **class balancing, outlier removal, and feature normalization** on a multiclass ADALINE classifier.

While the baseline achieved higher accuracy and AUC-ROC, it was heavily biased toward the majority class.

The balanced ADALINE model achieved:

```text
Accuracy:       75.48%
AUC-ROC:        87.95%
Macro Recall:   76%
Macro F1:       74%
```

The most important improvement was the increase in **Macro F1 from 0.49 to 0.74**, indicating substantially more balanced performance across the three classes.

This experiment therefore shows that:

> **A lower accuracy does not necessarily mean a worse classifier when the dataset is imbalanced.**

For this problem, **Macro F1 and Macro Recall provide a more meaningful evaluation of multiclass performance than accuracy alone.**

---

## Next Steps

Possible improvements for future experiments include:

* Hyperparameter tuning for ADALINE
* Learning-rate optimization
* Early stopping
* Different outlier detection methods
* Comparing ADALINE with Logistic Regression
* Comparing ADALINE with SVM
* Comparing against Tree-based models
* Feature selection
* PCA
* Cross-validation
* Calibration of OvR decision scores
* Evaluation using Precision-Recall curves
* Error analysis for misclassified samples

