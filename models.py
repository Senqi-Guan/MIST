from _model_registry import _MODEL_POOL

def get_model(model_name):
    model_pool = {'Metalearning Model': StackingClassifier(
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
    )}
    if model_name not in _MODEL_POOL:
        raise KeyError(f"Model {model_name} not found. Available: {list(_MODEL_POOL.keys())}")
    if model_name == "Metalearning Model":
        return model_pool[model_name]
    else:
        return _MODEL_POOL[model_name]
