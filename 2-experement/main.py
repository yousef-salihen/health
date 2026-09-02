from model.preprocessing import load_data, handle_missing_values, balance_dataset, detect_and_remove_outliers, z_score_normalization
from model.model import train_adaline, evaluate_model

def main():
    # load the dataset
    x_train, y_train, x_test, y_test = load_data()
    # clean the dataset
    x_train, y_train = balance_dataset(x_train, y_train)
    # fill in miss values
    x_train, x_test = handle_missing_values(x_train, x_test, strategy='mean')
    #detect and remove outliers
    x_train, y_train = detect_and_remove_outliers(x_train, y_train, threshold=1.5)
    # normalization
    x_train = z_score_normalization(x_train)
    x_test = z_score_normalization(x_test)
    # classification Adaline model
    adaline_model = train_adaline(x_train, y_train, learning_rate=0.01, n_iter=1000)
    adaline_auc = evaluate_model(adaline_model, x_train, y_train)
    print(f"Adaline AUC-ROC: {adaline_auc:.4f}")


if __name__ == "__main__":
    main()

