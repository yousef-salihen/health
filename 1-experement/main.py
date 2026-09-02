from model.preprocessing import load_data, handle_missing_values, check_correlation, apply_Zscoring, encode_labels
from model.model import train_logistic_regression, evaluate_model
def main():
    # load the dataset
    X_train, Y_train, X_test, Y_test = load_data()
    # clean the dataset
        #fill in miss values
    X_train, X_test = handle_missing_values(X_train, X_test, strategy = 'mean')
    # correlation Analysis
    corr_matrix = check_correlation(X_train, threshold = 0.8)
    # normalization 
    X_trainScaled, X_testScaled = apply_Zscoring(X_train, X_test)
    Y_trainEncoded = encode_labels(Y_train)
    Y_testEncoded = encode_labels(Y_test)
    results = {}
    #classification logestic
    lr_model = train_logistic_regression(X_trainScaled, Y_trainEncoded, penalty='l2', C=1.0)
    lr_auc = evaluate_model(lr_model, X_testScaled, Y_testEncoded)
    results['Logistic Regression'] = lr_auc
    print(f"Logistic Regression AUC-ROC !!!: {lr_auc :.4f}")
if __name__ == "__main__":
    main()