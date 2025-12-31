from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from fairlearn.metrics import true_positive_rate, false_positive_rate, selection_rate
import pandas as pd

def evaluate_models(X_train, y_train, X_test, y_test):
    models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "SVM": SVC(probability=True, random_state=42)
    }

    results = {}
    y_pred_results = {}
    y_proba_results = {}

    for name, model in models.items():
        pipe = Pipeline([
            ('classifier', model)
        ])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)[:, 1]
        y_pred_results[name] = y_pred
        y_proba_results[name] = y_proba

        results[name] = {
            "Accuracy": accuracy_score(y_test, y_pred),
            "F1": f1_score(y_test, y_pred),
            "ROC_AUC": roc_auc_score(y_test, y_proba),
            "true_positive_rate": true_positive_rate(y_test, y_pred),
            "false_positive_rate": false_positive_rate(y_test, y_pred),
            "selection_rate": selection_rate(y_test, y_pred)
        }

    results_df = pd.DataFrame(results).T
    return results_df, y_pred_results, y_proba_results, models

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    results = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "ROC_AUC": roc_auc_score(y_test, y_proba),
        "true_positive_rate": true_positive_rate(y_test, y_pred),
        "false_positive_rate": false_positive_rate(y_test, y_pred),
        "selection_rate": selection_rate(y_test, y_pred)
    }
    results_df = pd.DataFrame([results])

    return results_df, y_pred, y_proba

def weighted_crosstab(df, protected, target, weights):
    joint = (
        df
        .assign(w=weights)
        .groupby([protected, target])["w"]
        .sum()
        .unstack()
    )
    return joint / joint.values.sum()