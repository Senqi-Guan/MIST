from _model_registry import _MODEL_POOL

def get_model(model_name):
    if model_name not in _MODEL_POOL:
        raise KeyError(f"Model {model_name} not found. Available: {list(_MODEL_POOL.keys())}")
    return _MODEL_POOL[model_name]
