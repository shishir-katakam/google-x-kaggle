# src/imputation_tester.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error
from .tools import DataImputerTool
from typing import Dict, Any, Tuple, List

NUMERIC_STRATEGIES = ["mean", "median"]
CATEGORICAL_STRATEGIES = ["most_frequent", "constant"]

def _prepare_Xy(df: pd.DataFrame, target: str) -> Tuple[pd.DataFrame, pd.Series]:
    df2 = df.copy()
    y = df2[target]
    X = df2.drop(columns=[target])
    return X, y

def _encode_for_model(X_train: pd.DataFrame, X_val: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    X_train_enc = X_train.copy()
    X_val_enc = X_val.copy()
    for c in X_train_enc.columns:
        if X_train_enc[c].dtype == object or X_train_enc[c].dtype.name == "category":
            X_train_enc[c], uniques = pd.factorize(X_train_enc[c].astype(str))
            # Align categories in validation: map to codes, unknown -> -1
            cat_map = {v:i for i,v in enumerate(uniques)}
            X_val_enc[c] = X_val_enc[c].astype(str).map(cat_map).fillna(-1).astype(int)
    X_train_enc = X_train_enc.apply(pd.to_numeric, errors="coerce").fillna(0)
    X_val_enc = X_val_enc.apply(pd.to_numeric, errors="coerce").fillna(0)
    return X_train_enc, X_val_enc

def _score_imputation(df: pd.DataFrame, target: str, num_strategy: str, cat_strategy: str, problem_type: str) -> float:
    imputed_df, meta = DataImputerTool.impute(df, strategy_numeric=num_strategy, strategy_categorical=cat_strategy)
    X, y = _prepare_Xy(imputed_df, target)
    # require non-empty target
    if y.dropna().shape[0] < 30:
        raise ValueError("Not enough non-null target rows for reliable evaluation (need >=30).")
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=42)
    X_train_enc, X_val_enc = _encode_for_model(X_train, X_val)
    if problem_type == "classification":
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X_train_enc, y_train)
        preds = model.predict(X_val_enc)
        return float(accuracy_score(y_val, preds))
    else:
        model = RandomForestRegressor(n_estimators=50, random_state=42)
        model.fit(X_train_enc, y_train)
        preds = model.predict(X_val_enc)
        return float(-mean_squared_error(y_val, preds))

def find_best_imputation(df: pd.DataFrame, target: str, problem_type: str="classification", strategies_limit: int=None) -> Dict[str, Any]:
    results: List[Dict[str, Any]] = []
    nums = NUMERIC_STRATEGIES if strategies_limit is None else NUMERIC_STRATEGIES[:strategies_limit]
    cats = CATEGORICAL_STRATEGIES if strategies_limit is None else CATEGORICAL_STRATEGIES[:strategies_limit]
    for n in nums:
        for c in cats:
            try:
                score = _score_imputation(df, target, n, c, problem_type)
                results.append({"num_strategy": n, "cat_strategy": c, "score": score})
            except Exception as e:
                results.append({"num_strategy": n, "cat_strategy": c, "score": None, "error": str(e)})
    valid = [r for r in results if r.get("score") is not None]
    best = max(valid, key=lambda x: x["score"]) if valid else None
    return {"best": best, "results": results}
