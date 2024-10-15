import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import GradientBoostingClassifier
import shap
from sklearn.model_selection import KFold


# Load the data
data_path = "data/Features_used_as_input_of_ML_models_for_DF-Recognition_to_warnings.xlsx"
data = pd.read_excel(data_path)

# Split features and target variable. Indexing starts at 0, so slice columns 1:8 for features and column 9 for the target
X = data.iloc[:, 1:9]
y = data.iloc[:, 9]

# Use K-Fold Cross Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

for train_index, test_index in kf.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y[train_index], y[test_index]
    print("Train set class distribution:\n", y_train.value_counts())
    print("Test set class distribution:\n", y_test.value_counts())

# Initialize classifiers with fixed random seed
classifiers = {
    'SVM': make_pipeline(StandardScaler(), SVC(probability=True, random_state=42)),
    'kNN': KNeighborsClassifier(),
    'MLP': MLPClassifier(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
    'GBDT': GradientBoostingClassifier(random_state=42),
    'LightGBM': LGBMClassifier(random_state=42)
}

# Dictionary to store evaluation results
eval_results = {}

# Train and evaluate each classifier
for clf_name, clf in classifiers.items():
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1] if hasattr(clf, 'predict_proba') else None

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro', zero_division=1)
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')

    # Store evaluation metrics
    eval_results[clf_name] = {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}

# Print evaluation results for each classifier
for clf_name, metrics in eval_results.items():
    print(f"{clf_name}:")
    for metric_name, metric_value in metrics.items():
        print(f"\t{metric_name}: {metric_value:.2f}")

# Classification report for each classifier
for clf_name, clf in classifiers.items():
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # Generate classification report for each classifier
    report = classification_report(y_test, y_pred, output_dict=True)
    eval_results[clf_name] = report

    print(f"\n{clf_name} Classification Report:")
    for i in range(len(clf.classes_)):
        print(f"Class {i}:")
        print(f" Precision: {report[str(i)]['precision']:.2f}")
        print(f" Recall: {report[str(i)]['recall']:.2f}")
        print(f" F1-score: {report[str(i)]['f1-score']:.2f}")
        print(f" Support: {report[str(i)]['support']}")

    print(f"Overall Accuracy: {report['accuracy']:.2f}")

print("Unique target classes:", y.unique())
print("Target class distribution:\n", y.value_counts())

print("Predicted class distribution:\n", pd.Series(y_pred).value_counts())
print("Actual class distribution:\n", y_test.value_counts())

print(X.head())
print(y.head())

# SHAP Analysis for Random Forest
rf_clf = RandomForestClassifier(random_state=42)
rf_clf.fit(X_train, y_train)

# Create SHAP explainer for Random Forest
rf_explainer = shap.Explainer(rf_clf, X_train, check_additivity=False)
rf_shap_values = rf_explainer(X_test, check_additivity=False)

# Plot SHAP summary plot for each class
num_classes = len(rf_clf.classes_)
print(f"Number of classes: {num_classes}")

for i in range(num_classes):
    rf_shap_values_for_class_i = rf_shap_values[..., i]
    print(f"Class {i} SHAP Summary Plot")
    shap.summary_plot(rf_shap_values_for_class_i, X_test, plot_type="dot")
