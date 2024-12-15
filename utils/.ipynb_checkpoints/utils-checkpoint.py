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

