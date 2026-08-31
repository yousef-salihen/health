import pandas as pd
import numpy as np
#=========================================================================================================================================
#load dataset
#=========================================================================================================================================
def load_data():
    # Load the dataset from CSV files
    train_df = pd.read_csv('/Users/yousefsalihen/Study/full_pipline/health_condition/dataset/train.csv')
    
    X_train = train_df.drop(columns=['id', 'health_condition']).values
    Y_train = train_df['health_condition'].values
    test_df = pd.read_csv('/Users/yousefsalihen/Study/full_pipline/health_condition/dataset/test.csv')
    X_test = test_df.drop(columns="id").values
    test = pd.read_csv('/Users/yousefsalihen/Study/full_pipline/health_condition/dataset/sample_submission.csv').iloc[:, 1]
    Y_test = test.values

    print(f"training data : {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"testing data : {X_test.shape[0]} samples, {X_test.shape[1]} features")

    return X_train, Y_train, X_test, Y_test
#=========================================================================================================================================
#sperate numeric and categorical features and combine them back together
#=========================================================================================================================================
def seprete_numeric_categorical(X):
    #seprete the numeric and categorical features
    x_numeric = X[:, :7].astype(float)
    x_categorical = X[:, 7:]
    return x_numeric, x_categorical
def combine_numeric_categorical(x_numeric, x_categorical):
    #combine the numeric and categorical features back together
    X = np.hstack((x_numeric, x_categorical))
    return X
#=========================================================================================================================================
#handle missing values
#=========================================================================================================================================
def handle_missing_values(X_train, X_test, strategy='mean'):
    #seprete the numeric and categorical features
    x_numeric_train, x_categorical_train = seprete_numeric_categorical(X_train)
    x_numeric_test, x_categorical_test = seprete_numeric_categorical(X_test)
    #fill in missing values for numeric features
    if strategy == 'mean':
        mean_values = np.nanmean(x_numeric_train, axis=0)
        x_numeric_train = np.where(np.isnan(x_numeric_train), mean_values, x_numeric_train)
        x_numeric_test = np.where(np.isnan(x_numeric_test), mean_values, x_numeric_test)
    elif strategy == 'median':
        median_values = np.nanmedian(x_numeric_train, axis=0)
        x_numeric_train = np.where(np.isnan(x_numeric_train), median_values, x_numeric_train)
        x_numeric_test = np.where(np.isnan(x_numeric_test), median_values, x_numeric_test)
        
    #fill in missing values for categorical features
    for i in range(x_categorical_train.shape[1]):
        mode_value = pd.Series(x_categorical_train[:, i]).mode()[0]
        x_categorical_train[:, i] = np.where(pd.isnull(x_categorical_train[:, i]), mode_value, x_categorical_train[:, i])
        x_categorical_test[:, i] = np.where(pd.isnull(x_categorical_test[:, i]), mode_value, x_categorical_test[:, i])
    #combine the numeric and categorical features back together
    X_train = np.hstack((x_numeric_train, x_categorical_train))
    X_test = np.hstack((x_numeric_test, x_categorical_test))
    return X_train, X_test
#==========================================================================================================================================
#correlation analysis
#==========================================================================================================================================
def check_correlation(X, threshold = 0.8 ):
    print('\nCheching the correlation')
    X_sperete, _ = seprete_numeric_categorical(X)
    #calculate correlation matrix
    corr_matrix = np.corrcoef(X_sperete.T)

    #find high correlation pairs
    high_corr_pair = []
    n_features = X.shape[1]
    for i in range(corr_matrix.shape[0]):
        for j in range(i+1, corr_matrix.shape[1]):
            if abs(corr_matrix[i, j]) > threshold:
                high_corr_pair.append((i, j, corr_matrix[i, j]))
    if len(high_corr_pair) == 0:
        print(f'Found {len(high_corr_pair)} pairs of highly correlated features (threshold = {threshold}):')
    else:
        print(f'Found {len(high_corr_pair)} pairs of highly correlated features (threshold = {threshold}):')
        for i, j, corr in high_corr_pair:
            print(f'Feature {i} and Feature {j}: correlation = {corr:.2f}')
    return corr_matrix
#==========================================================================================================================================
#probability encoding categorical features
#==========================================================================================================================================
def encode_categorical_features(X_categorical_train, X_categorical_test):
    print('\nEncoding categorical features')
    # Create a dictionary to store the mapping for each categorical feature
    encoding_dict = {}
    for i in range(X_categorical_train.shape[1]):
        # Calculate the probability of each category in the training set
        category_counts = pd.Series(X_categorical_train[:, i]).value_counts()
        total_count = len(X_categorical_train)
        category_probabilities = category_counts / total_count

        # Store the mapping in the dictionary
        encoding_dict[i] = category_probabilities.to_dict()

        # Replace categories with their probabilities in the training set
        X_categorical_train[:, i] = [encoding_dict[i].get(cat, 0) for cat in X_categorical_train[:, i]]

        # Replace categories with their probabilities in the test set
        X_categorical_test[:, i] = [encoding_dict[i].get(cat, 0) for cat in X_categorical_test[:, i]]

    return X_categorical_train.astype(float), X_categorical_test.astype(float)

#encode the labels
def encode_labels(Y):
    print('\nEncoding labels')
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    Y_encoded = le.fit_transform(Y)
    return Y_encoded
#==========================================================================================================================================
#Z-score normalization
#==========================================================================================================================================
def apply_Zscoring (X_train, X_test):
    print('\napply Z-score normlization')
    # separate the numeric and categorical features
    x_numeric_train, X_categorical_train = seprete_numeric_categorical(X_train)
    x_numeric_test, X_categorical_test = seprete_numeric_categorical(X_test)
    mean = np.mean(x_numeric_train, axis=0)
    std = np.std(x_numeric_train, axis=0)
    std[std == 0] = 1  # Avoid division by zero for constant features

    X_train = (x_numeric_train - mean) / std
    X_test = (x_numeric_test - mean) / std
    X_categorical_train, X_categorical_test = encode_categorical_features(X_categorical_train, X_categorical_test)
    X_train_scaled = combine_numeric_categorical(X_train, X_categorical_train)
    X_test_scaled = combine_numeric_categorical(X_test, X_categorical_test)

    return X_train_scaled, X_test_scaled