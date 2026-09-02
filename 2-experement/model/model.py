import numpy as np

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)


# ========================================================================
# ADALINE
# ========================================================================

class Adaline:

    def __init__(self, learning_rate=0.01, n_iter=1000):
        self.learning_rate = learning_rate
        self.n_iter = n_iter

        self.weights = None
        self.bias = 0.0

    # --------------------------------------------------------------------
    # Training
    # --------------------------------------------------------------------

    def fit(self, X, y):

        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        n_samples, n_features = X.shape

        # Initialize weights and bias
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        # Gradient Descent
        for _ in range(self.n_iter):

            # Linear output
            output = np.dot(X, self.weights) + self.bias

            # Error
            errors = y - output

            # Gradient of weights
            dw = -(2 / n_samples) * np.dot(X.T, errors)

            # Gradient of bias
            db = -(2 / n_samples) * np.sum(errors)

            # Update weights
            self.weights -= self.learning_rate * dw

            # Update bias
            self.bias -= self.learning_rate * db

        return self

    # --------------------------------------------------------------------
    # Decision Function
    # --------------------------------------------------------------------

    def decision_function(self, X):

        X = np.asarray(X, dtype=float)

        return np.dot(X, self.weights) + self.bias


# ========================================================================
# MULTI-CLASS ADALINE
# One-vs-Rest
# ========================================================================

class MultiClassAdaline:

    def __init__(self, learning_rate=0.01, n_iter=1000):

        self.learning_rate = learning_rate
        self.n_iter = n_iter

        self.models = {}
        self.classes_ = None

    # --------------------------------------------------------------------
    # Training
    # --------------------------------------------------------------------

    def fit(self, X, y):

        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        # Get unique classes
        self.classes_ = np.unique(y)

        print(f"Classes: {self.classes_}")

        # Train one ADALINE for every class
        for cls in self.classes_:

            print(f"Training ADALINE for class: {cls}")

            # One-vs-Rest target
            #
            # Current class -> 1
            # Other classes -> 0
            y_binary = np.where(y == cls, 1, 0)

            model = Adaline(
                learning_rate=self.learning_rate,
                n_iter=self.n_iter
            )

            model.fit(X, y_binary)

            self.models[cls] = model

        return self

    # --------------------------------------------------------------------
    # Decision Function
    # --------------------------------------------------------------------

    def decision_function(self, X):

        scores = []

        # Get score from every binary ADALINE
        for cls in self.classes_:

            score = self.models[cls].decision_function(X)

            scores.append(score)

        # Shape:
        #
        # (number_of_samples, number_of_classes)
        #
        return np.column_stack(scores)

    # --------------------------------------------------------------------
    # Prediction
    # --------------------------------------------------------------------

    def predict(self, X):

        scores = self.decision_function(X)

        # Select class with highest score
        class_index = np.argmax(scores, axis=1)

        return self.classes_[class_index]


# ========================================================================
# TRAIN ADALINE
# ========================================================================

def train_adaline(
    X_train,
    Y_train,
    learning_rate=0.01,
    n_iter=1000
):

    print("\n" + "=" * 70)
    print("Training ADALINE model")
    print("=" * 70)

    adaline = MultiClassAdaline(
        learning_rate=learning_rate,
        n_iter=n_iter
    )

    adaline.fit(X_train, Y_train)

    print("ADALINE training completed.")

    return adaline


# ========================================================================
# SOFTMAX
# Used to convert ADALINE scores into probability-like values
# ========================================================================

def softmax(scores):

    scores = np.asarray(scores)

    # Numerical stability
    scores = scores - np.max(
        scores,
        axis=1,
        keepdims=True
    )

    exp_scores = np.exp(scores)

    probabilities = exp_scores / np.sum(
        exp_scores,
        axis=1,
        keepdims=True
    )

    return probabilities


# ========================================================================
# EVALUATE MODEL
# ========================================================================

def evaluate_model(model, X, Y):

    print("\n" + "=" * 70)
    print("Evaluating ADALINE model")
    print("=" * 70)

    # --------------------------------------------------------------------
    # Prediction
    # --------------------------------------------------------------------

    Y_pred = model.predict(X)

    # --------------------------------------------------------------------
    # Decision scores
    # --------------------------------------------------------------------

    Y_scores = model.decision_function(X)

    # --------------------------------------------------------------------
    # Convert scores to probabilities
    # --------------------------------------------------------------------

    Y_proba = softmax(Y_scores)

    # --------------------------------------------------------------------
    # Accuracy
    # --------------------------------------------------------------------

    accuracy = accuracy_score(Y, Y_pred)

    print(f"\nAccuracy: {accuracy:.4f}")

    # --------------------------------------------------------------------
    # Classification Report
    # --------------------------------------------------------------------

    print("\nClassification Report:")

    print(
        classification_report(
            Y,
            Y_pred
        )
    )

    # --------------------------------------------------------------------
    # Confusion Matrix
    # --------------------------------------------------------------------

    print("\nConfusion Matrix:")

    print(
        confusion_matrix(
            Y,
            Y_pred
        )
    )

    # --------------------------------------------------------------------
    # ROC-AUC
    # --------------------------------------------------------------------

    auc_score = roc_auc_score(
        Y,
        Y_proba,
        multi_class="ovr",
        average="macro"
    )

    print(f"\nAUC-ROC (OvR): {auc_score:.4f}")

    # --------------------------------------------------------------------
    # Return
    # --------------------------------------------------------------------

    return auc_score