from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import ExtraTreesClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.ensemble import StackingClassifier

_MODEL_POOL = {
    'lr': LogisticRegression(
        random_state=42,
        max_iter=50,
        C=0.01,
        class_weight='balanced'
    ),
    'svm': SVC(kernel='rbf', probability=True, random_state=42, C=0.25, gamma='scale', class_weight='balanced'),
    'randomforest': RandomForestClassifier(
        n_estimators=400,
        max_depth=3,
        min_samples_split=10,
        max_features=0.2,
        random_state=42,
        class_weight='balanced'
    ),
    'extratrees': ExtraTreesClassifier(
        n_estimators=1000,
        max_depth=4,
        min_samples_leaf=5,
        min_samples_split=10,
        max_features=0.35,
        min_impurity_decrease=0.0,
        bootstrap=True,
        oob_score=True,
        criterion='entropy',
        random_state=42,
        n_jobs=-1
    ),
    'xgboost': XGBClassifier(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=4,
        colsample_bytree=0.9,
        subsample=1.0,
        random_state=42,
        eval_metric='logloss',
    ),
    'LightGBM': LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        num_leaves=4,
        random_state=42,
        class_weight='balanced',
        verbose=-1,
        feature_fraction=1.0,
        bagging_fraction=1.0
    ),
    'Metalearning Model': StackingClassifier(
        estimators=[
            ('lr', LogisticRegression(random_state=42, max_iter=500, C=0.3, class_weight='balanced')),
            ('lightgbm', LGBMClassifier(n_estimators=200, learning_rate=0.05, max_depth=4, num_leaves=4, random_state=42, class_weight='balanced', verbose=-1, feature_fraction=1, bagging_fraction=1)),
            ('SVM', SVC(kernel='rbf', probability=True, random_state=42, C=0.8, gamma='scale', class_weight='balanced')),
        ],
        final_estimator=LogisticRegression(random_state=42, max_iter=2000, C=0.5, class_weight='balanced'),
        cv=5,
        stack_method='predict_proba',
        n_jobs=-1,
        passthrough=True,
    )
}
