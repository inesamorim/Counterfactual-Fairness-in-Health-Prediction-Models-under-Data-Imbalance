from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from fairlearn.metrics import MetricFrame, selection_rate, true_positive_rate, false_positive_rate
from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference

def display_fairness_metrics(X_test, y_test, y_pred, sensitive_feature='sex'):
    sensitive = X_test[sensitive_feature]

    metrics = {
        'accuracy': accuracy_score,
        'selection_rate': selection_rate,
        'tpr': true_positive_rate,
        'fpr': false_positive_rate
    }

    frame = MetricFrame(metrics=metrics, y_true=y_test, y_pred=y_pred, sensitive_features=sensitive)
    print(frame.by_group)

    dp_diff = demographic_parity_difference(y_true=y_test, y_pred=y_pred, sensitive_features=sensitive)
    eo_diff = equalized_odds_difference(y_true=y_test, y_pred=y_pred, sensitive_features=sensitive)

    print("\n# Fairness Summary:")
    print(f"Demographic Parity Difference: {dp_diff:.4f}")
    print(f"Equalized Odds Difference: {eo_diff:.4f}")
