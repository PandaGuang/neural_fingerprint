from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

def calculate_metrics(y_true, y_pred):
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    # For binary classification, AUC and ROC can be computed if probabilities are available
    # Here, assuming binary classification for simplicity
    try:
        auc = roc_auc_score(y_true, y_pred)
    except ValueError:
        auc = None  # AUC cannot be computed if only one class is present
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics = {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'confusion_matrix': {'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn}
    }
    return metrics
