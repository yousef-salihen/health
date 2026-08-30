# Health Condition Classification

## Overview

This repository contains a systematic machine learning study for **health condition classification**.

The project is organized as a sequence of experiments. Each experiment introduces a specific modeling, preprocessing, feature engineering, or optimization technique and evaluates its effect on classification performance.

The main objective is to build a reliable machine learning pipeline while understanding **how different techniques affect model performance**.

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
├── 1-experement/
│   ├── README.md
│   ├── main.py
│   └── model/
│       ├── model.py
│       └── preprocessing.py
│
├── 2-experement/
│   └── ...
│
├── 3-experement/
│   └── ...
│
└── README.md
```

Each experiment contains its own implementation and documentation.

---

# Dataset

The dataset contains **13 input features** and one target variable.

The `id` column is used only as an identifier and is not used as a model feature.

## Features

| Feature                   | Type        | Description                |
| ------------------------- | ----------- | -------------------------- |
| `sleep_duration`          | Numerical   | Sleep duration             |
| `heart_rate`              | Numerical   | Heart rate                 |
| `bmi`                     | Numerical   | Body Mass Index            |
| `calorie_expenditure`     | Numerical   | Calorie expenditure        |
| `step_count`              | Numerical   | Number of steps            |
| `exercise_duration`       | Numerical   | Exercise duration          |
| `water_intake`            | Numerical   | Water intake               |
| `diet_type`               | Categorical | Type of diet               |
| `stress_level`            | Categorical | Stress-level category      |
| `sleep_quality`           | Categorical | Sleep-quality category     |
| `physical_activity_level` | Categorical | Physical-activity category |
| `smoking_alcohol`         | Categorical | Smoking/alcohol category   |
| `gender`                  | Categorical | Gender category            |

### Target

```text
health_condition
```

The target contains **three classes**:

```text
Class 0
Class 1
Class 2
```

---

## Feature Groups

The input features are divided into numerical and categorical features.

### Numerical Features

```text
sleep_duration
heart_rate
bmi
calorie_expenditure
step_count
exercise_duration
water_intake
```

### Categorical Features

```text
diet_type
stress_level
sleep_quality
physical_activity_level
smoking_alcohol
gender
```

This separation is used during preprocessing because numerical and categorical features require different transformations.

---

# Experimental Approach

The project follows an iterative experimental methodology:

```text
                    Dataset
                       │
                       ▼
                Data Preprocessing
                       │
                       ▼
                Feature Analysis
                       │
                       ▼
                  Baseline
                       │
                       ▼
                 Experiment
                       │
                       ▼
                  Evaluation
                       │
                       ▼
              Analyze Results
                       │
                       ▼
             Improve / Modify
                       │
                       ▼
              Next Experiment
```

Each experiment is designed to answer a specific question rather than simply train another model.

---

# General Pipeline

The experiments may include different combinations of the following stages:

### 1. Data Loading

Load the training and testing datasets and separate:

* Input features
* Target labels
* Identifiers

### 2. Data Cleaning

Handle issues such as:

* Missing numerical values
* Missing categorical values
* Invalid values
* Data type conversion

### 3. Feature Analysis

Investigate the input features using techniques such as:

* Correlation analysis
* Feature distributions
* Feature relationships
* Feature selection

### 4. Feature Transformation

Depending on the experiment, different transformations may be evaluated:

* Z-score normalization
* Categorical encoding
* Feature selection
* Other feature engineering techniques

### 5. Model Training

Different machine learning algorithms can be evaluated against the baseline.

The first experiment uses **Logistic Regression** as the baseline classifier.

### 6. Evaluation

Models are evaluated using multiple metrics rather than relying on accuracy alone.

Main evaluation metrics include:

* Accuracy
* Precision
* Recall
* F1-score
* Macro F1-score
* Confusion Matrix
* ROC-AUC

For multiclass classification, ROC-AUC is evaluated using the **One-vs-Rest (OvR)** approach.

---

# Experiments

## Experiment 1 — Logistic Regression Baseline

The first experiment establishes a simple baseline using:

```text
Missing Value Handling
        ↓
Correlation Analysis
        ↓
Z-Score Normalization
        ↓
Categorical Feature Encoding
        ↓
Label Encoding
        ↓
Logistic Regression
        ↓
Model Evaluation
```

### Model

```text
Algorithm: Logistic Regression
Penalty: L2
C: 1.0
Solver: liblinear
Random State: 42
```

### Baseline Results

```text
Accuracy: 87.16%
ROC-AUC: 90.54%
Macro F1: 0.49
```

The baseline achieves strong overall performance but shows a significant difference in performance between the three classes.

In particular, the minority classes have substantially lower recall than the majority class.

See [`1-experement/README.md`](1-experement/README.md) for the detailed implementation and analysis.

---

# Class Distribution

The initial dataset contains three target classes with an imbalanced distribution:

| Class     |     Samples | Approx. Distribution |
| --------- | ----------: | -------------------: |
| 0         |     592,561 |                85.9% |
| 1         |      39,803 |                 5.8% |
| 2         |      57,724 |                 8.4% |
| **Total** | **690,088** |             **100%** |

The large difference between the classes is an important consideration when interpreting model performance.

A high accuracy score may not necessarily indicate good performance across all classes.

Therefore, future experiments will pay particular attention to:

* Minority-class recall
* Macro F1
* Per-class F1
* Confusion Matrix
* ROC-AUC

---

# Experimental Questions

Future experiments will investigate questions such as:

## Class Imbalance

* Does undersampling the majority class improve minority-class performance?
* How much information is lost through undersampling?
* Is class weighting more effective than removing samples?

## Regularization

* How does L1 regularization compare with L2?
* Which features are eliminated by L1 regularization?
* How does regularization strength affect performance?

## Hyperparameter Optimization

* How does changing `C` affect Logistic Regression?
* Which configuration provides the best balance between performance and complexity?

## Feature Engineering

* Do different encoding techniques improve classification?
* Does removing highly correlated features improve performance?
* Which features contribute most to the classification task?

## Model Comparison

After establishing strong baselines, nonlinear models can be evaluated and compared against Logistic Regression.

---

# Evaluation Strategy

Because the dataset is highly imbalanced, model comparison should not rely on accuracy alone.

The primary comparison metrics are:

```text
Macro F1
Per-Class Recall
Per-Class F1
ROC-AUC
Confusion Matrix
Accuracy
```

A model should be considered an improvement when it provides a meaningful improvement in the target metrics without creating an unacceptable degradation in other important metrics.

---

# Experiment Tracking

Each experiment should document:

```text
Experiment Number
Objective
Dataset Configuration
Preprocessing
Feature Engineering
Model
Hyperparameters
Evaluation Metrics
Results
Observations
Limitations
Next Step
```

This makes it possible to understand not only **which model performs best**, but also **why the performance changed** between experiments.

---

# Reproducibility

Experiments should use fixed random seeds where applicable.

The baseline Logistic Regression experiment currently uses:

```python
random_state=42
```

The same experimental conditions should be maintained when comparing models whenever possible.

---

# Current Status

| Experiment | Method                           | Status      |
| ---------- | -------------------------------- | ----------- |
| 1          | Logistic Regression Baseline     | ✅ Completed |
| 2          | Class Imbalance                  | 🔄 Planned  |
| 3          | Regularization / Hyperparameters | 🔄 Planned  |
| 4          | Feature Selection                | 🔄 Planned  |
| 5          | Nonlinear Models                 | 🔄 Planned  |
| 6          | Model Comparison                 | 🔄 Planned  |

---

# Goal

The final objective of this repository is to develop a **systematic and interpretable machine learning pipeline** rather than optimizing a single metric.

The experiments will progressively investigate:

```text
Data
 ↓
Preprocessing
 ↓
Feature Analysis
 ↓
Baseline
 ↓
Class Imbalance
 ↓
Feature Engineering
 ↓
Regularization
 ↓
Hyperparameter Optimization
 ↓
Model Comparison
 ↓
Final Model
```

Each experiment should provide evidence that supports the decision made in the next stage of the pipeline.
