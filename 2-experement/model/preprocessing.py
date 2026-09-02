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
#==========================================================================================================================================
#probability encoding categorical features
#==========================================================================================================================================
def encode_categorical_features(X_categorical):
    print('\nEncoding categorical features')
    # Create a dictionary to store the mapping for each categorical feature
    encoding_dict = {}
    for i in range(X_categorical.shape[1]):
        # Calculate the probability of each category in the training set
        category_counts = pd.Series(X_categorical[:, i]).value_counts()
        total_count = len(X_categorical)
        category_probabilities = category_counts / total_count

        # Store the mapping in the dictionary
        encoding_dict[i] = category_probabilities.to_dict()

        # Replace categories with their probabilities in the training set
        X_categorical[:, i] = [encoding_dict[i].get(cat, 0) for cat in X_categorical[:, i]]


    return X_categorical.astype(float)
#encode the labels
def encode_labels(Y):
    print('\nEncoding labels')
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    Y_encoded = le.fit_transform(Y)
    return Y_encoded
#=========================================================================================================================================
# balance the dataset by removing the samples from the majority class to match the number of samples in the minority class
#=========================================================================================================================================
def balance_dataset(X_train, Y_train):
    #seperate the numeric and categorical features
    x_train_numeric, x_train_categorical = seprete_numeric_categorical(X_train)
    # encoding the categorical features using probability encoding
    x_train_categorical = encode_categorical_features(x_train_categorical)
    #combine the numeric and categorical features back together
    X_train = np.hstack((x_train_numeric, x_train_categorical))
    # encode the labels using label encoding
    Y_train = encode_labels(Y_train)
    # Count the number of samples in each class
    class_counts = np.bincount(Y_train)
    min_class_count = np.min(class_counts[class_counts > 0])  # Minimum count of non-zero classes

    # Create a mask to select samples from each class
    mask = np.zeros_like(Y_train, dtype=bool)
    for class_label in np.unique(Y_train):
        class_mask = (Y_train == class_label)
        selected_indices = np.random.choice(np.where(class_mask)[0], min_class_count, replace=False)
        mask[selected_indices] = True

    # Apply the mask to balance the dataset
    X_train_balanced = X_train[mask]
    Y_train_balanced = Y_train[mask]

    print(f"Balanced training data : {X_train_balanced.shape[0]} samples, {X_train_balanced.shape[1]} features")
    
    return X_train_balanced, Y_train_balanced

#=========================================================================================================================================
#handle missing values
#=========================================================================================================================================
# categorical handling with KNN interpretre
def knn_for_categorical (x_train, x_test):
    # Create a KNN imputer for categorical features
    x_train = encode_categorical_features(x_train)
    x_test = encode_categorical_features(x_test)
    from sklearn.impute import KNNImputer
    knn_imputer = KNNImputer(n_neighbors=5, weights="uniform", missing_values=np.nan)
    # Fit the imputer on the training data and transform both training and test data
    x_train_imputed = knn_imputer.fit_transform(x_train)
    x_test_imputed = knn_imputer.transform(x_test)
    return x_train_imputed, x_test_imputed

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
    x_categorical_train, x_categorical_test = knn_for_categorical(x_categorical_train, x_categorical_test)
    #combine the numeric and categorical features back together
    X_train = np.hstack((x_numeric_train, x_categorical_train))
    X_test = np.hstack((x_numeric_test, x_categorical_test))
    return X_train, X_test
#=========================================================================================================================================
#detect and remove outliers using third quartile and interquartile range (IQR)
#=========================================================================================================================================
def detect_and_remove_outliers(X_train, Y_train, threshold=1.5):
    print('\nDetecting and removing outliers')
    #seperate the numeric and categorical features
    x_numeric_train, x_categorical_train = seprete_numeric_categorical(X_train)
    # Calculate the first (Q1) and third (Q3) quartiles for each numeric feature
    Q1 = np.percentile(x_numeric_train, 25, axis=0)
    Q3 = np.percentile(x_numeric_train, 75, axis=0)
    IQR = Q3 - Q1

    # Define the lower and upper bounds for outlier detection
    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR

    # Create a mask to identify non-outlier samples
    non_outlier_mask = np.all((x_numeric_train >= lower_bound) & (x_numeric_train <= upper_bound), axis=1)

    # Apply the mask to remove outliers from both features and labels
    X_train_no_outliers = X_train[non_outlier_mask]
    Y_train_no_outliers = Y_train[non_outlier_mask]

    print(f"Training data after removing outliers: {X_train_no_outliers.shape[0]} samples, {X_train_no_outliers.shape[1]} features")
    
    return X_train_no_outliers, Y_train_no_outliers
#==========================================================================================================================================
#z-score normalization
#==========================================================================================================================================
def z_score_normalization(X):
    print('\nApplying z-score normalization')
    #seperate the numeric and categorical features
    x_numeric, x_categorical = seprete_numeric_categorical(X)
    # Calculate the mean and standard deviation for each numeric feature
    mean = np.mean(x_numeric, axis=0)
    std_dev = np.std(x_numeric, axis=0)

    # Apply z-score normalization to numeric features
    x_numeric_normalized = (x_numeric - mean) / std_dev

    #combine the numeric and categorical features back together
    X_normalized = np.hstack((x_numeric_normalized, x_categorical))
    
    return X_normalized
