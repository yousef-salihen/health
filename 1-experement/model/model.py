import numpy as np
#=========================================================================================================================================
# train logistic regression model
#=========================================================================================================================================
def train_logistic_regression(X_train, Y_train, penalty='l2', C=1.0):
    print("Training Logistic Regression model (penalty: {}, C: {})...".format(penalty, C))
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(
        penalty=penalty,
        C=C,
        solver='liblinear',
        random_state=42
    )
    model.fit(X_train, Y_train)
    # count non-zero coefficients (for L1)
    if penalty == 'l1':
        non_zero_coefficients = np.sum(model.coef_ != 0)
        print(f"Number of non-zero coefficients: {non_zero_coefficients}")
    print("Model training completed.")
    return model
#=========================================================================================================================================
# evaluate the model
#=========================================================================================================================================
def evaluate_model(model, X, Y):
    print("Evaluating model...")

    Y_pred = model.predict(X)
    Y_pred_proba = model.predict_proba(X)

    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        roc_auc_score
    )

    # Accuracy
    accuracy = accuracy_score(Y, Y_pred)
    print(f"Accuracy: {accuracy:.4f}")

    # Classification report
    print("\nClassification Report:")
    print(classification_report(Y, Y_pred))

    # Confusion Matrix
    print("\nConfusion Matrix:")
    print(confusion_matrix(Y, Y_pred))

    # Multi-class AUC
    auc_score = roc_auc_score(
        Y,
        Y_pred_proba,
        multi_class='ovr',
        average='macro'
    )

    print(f"\nAUC-ROC (OvR): {auc_score:.4f}")

    return auc_score