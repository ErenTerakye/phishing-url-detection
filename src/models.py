import time
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier


def get_baseline_models():
    return {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        'XGBoost': XGBClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1,
            eval_metric='logloss',
            verbosity=0,
        ),
        'SVM': SVC(kernel='rbf', probability=True, random_state=42),
        'KNN': KNeighborsClassifier(n_neighbors=5),
    }


def train_model(model, X_train, y_train):
    start = time.time()
    model.fit(X_train, y_train)
    elapsed = time.time() - start
    print(f"  Training time: {elapsed:.2f}s")
    return model, elapsed


def get_hyperparameter_grids():
    return {
        'Logistic Regression': {
            'C': [0.01, 0.1, 1, 10],
            'solver': ['lbfgs', 'liblinear'],
        },
        'Decision Tree': {
            'max_depth': [5, 10, 20, None],
            'min_samples_split': [2, 5, 10],
        },
        'Random Forest': {
            'n_estimators': [100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5],
        },
        'XGBoost': {
            'n_estimators': [100, 200],
            'max_depth': [3, 6, 10],
            'learning_rate': [0.01, 0.1, 0.3],
        },
        'KNN': {
            'n_neighbors': [3, 5, 7, 11],
            'weights': ['uniform', 'distance'],
        },
    }
