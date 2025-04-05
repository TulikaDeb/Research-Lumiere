from sklearn.metrics import accuracy_score,recall_score, f1_score, precision_score, balanced_accuracy_score


def print_metrics(y_true, y_pred):
    '''
    Functions that prints and returns the accuracy, balance, precision, recall, f1
    metrics from sklearn.metrics, when given the true labels and the predcited ones. 
    ----
    Input:
        y_test (np.array): test labels
        y_pred (np.array): predicted labels
    Output:
        accuracy (float): sklearn accuracy_score
        balance (float): sklearn balanced_accuracy_score
        precision (float): sklearn precision_score
        recall (float): sklearn recall_score
        f1 (float): sklearn f1_score
    '''
    accuracy = accuracy_score(y_true, y_pred)
    balance = balanced_accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')  
    recall = recall_score(y_true, y_pred, average='weighted')        
    f1 = f1_score(y_true, y_pred, average='weighted')

    print(f"Accuracy: {accuracy:.2f}")
    print(f"Balanced Accuracy: {balance:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")

    return accuracy, balance, precision, recall, f1



import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

def plot_confusion_matrix(y_test, y_pred):
    '''
    Plot confusion matrix
    '''
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(9), yticklabels=range(9))
    plt.title('Confusion Matrix for 9-Class Classification')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.show()





import pandas as pd
from sklearn.model_selection import train_test_split
def dataset_traintest_split(df, target_column):
    '''
    Splits a DataFrame into training and testing sets for features and target.
    
    Parameters:
    - df (pd.DataFrame): The input DataFrame containing features and target.
    - target_columns (list or str): The name(s) of the target column(s) in the DataFrame.
    
    Returns:
    - x_train (pd.DataFrame): Training features.
    - x_test (pd.DataFrame): Testing features.
    - y_train (pd.Series): Training target.
    - y_test (pd.Series): Testing target.
    '''

    x = df.drop(columns=target_column)
    y = df[target_column]
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    
    return x_train, x_test, y_train, y_test


import pandas as pd

def manual_kfold_split(x, y):
    """
    Splits the dataset into k folds for cross-validation.

    Parameters:
    x (pd.DataFrame): Feature dataset.
    y (pd.Series): Target variable.
    k (int): Number of folds.

    Returns:
    list: A list of tuples, each containing (x_train, y_train, x_test, y_test) for each fold.
    """
    folds = []
    fold_size = len(x) // 5
    
    for i in range(k):
        x_test2 = x.iloc[i * fold_size : (i + 1) * fold_size]
        y_test2 = y.iloc[i * fold_size : (i + 1) * fold_size]
        x_train2 = pd.concat([x.iloc[:i * fold_size], x.iloc[(i + 1) * fold_size:]])
        y_train2 = pd.concat([y.iloc[:i * fold_size], y.iloc[(i + 1) * fold_size:]])
        
        folds.append((x_train, y_train, x_test, y_test))
    
    return folds


