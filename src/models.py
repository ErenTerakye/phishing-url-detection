import time
import warnings
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import PassiveAggressiveClassifier, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

try:
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover - depends on local environment
    XGBClassifier = None


def get_baseline_models():
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(
            n_estimators=120,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced_subsample',
        ),
        # LinearSVC is used instead of RBF SVC because PhiUSIIL has 235k rows.
        # It keeps the SVM comparison reproducible while avoiding impractical runtime.
        'SVM': LinearSVC(random_state=42, class_weight='balanced', max_iter=5000),
        'KNN': KNeighborsClassifier(n_neighbors=5),
    }
    if XGBClassifier is not None:
        models['XGBoost'] = XGBClassifier(
            n_estimators=150,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1,
            eval_metric='logloss',
            tree_method='hist',
            verbosity=0,
        )
    else:
        warnings.warn("xgboost kurulu degil; XGBoost modeli atlanacak.", RuntimeWarning)
    return models


def build_models(random_state=42):
    """Build the final-project model set with graceful XGBoost fallback."""
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=random_state,
            class_weight="balanced",
        ),
        "Decision Tree": DecisionTreeClassifier(
            random_state=random_state,
            class_weight="balanced",
            max_depth=None,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=150,
            random_state=random_state,
            n_jobs=-1,
            class_weight="balanced_subsample",
        ),
        "SVM": LinearSVC(
            random_state=random_state,
            class_weight="balanced",
            max_iter=5000,
        ),
        "GaussianNB": GaussianNB(),
        "SGDClassifier": SGDClassifier(
            loss="log_loss",
            random_state=random_state,
            class_weight="balanced",
            max_iter=1000,
            tol=1e-3,
        ),
    }

    if XGBClassifier is not None:
        models["XGBoost"] = XGBClassifier(
            n_estimators=180,
            max_depth=6,
            learning_rate=0.08,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=random_state,
            n_jobs=-1,
            eval_metric="logloss",
            tree_method="hist",
            verbosity=0,
        )
    else:
        warnings.warn("xgboost kurulu degil; XGBoost modeli atlandi.", RuntimeWarning)

    return models


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
