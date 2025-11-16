import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from typing import Dict, Any, List, Tuple


class SchemaInferTool:
    @staticmethod
    def infer(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        schema = {}
        for col in df.columns:
            ser = df[col]
            dtype = str(ser.dtype)
            non_null_count = ser.notna().sum()
            pct_null = 1 - non_null_count / max(1, len(ser))
            schema[col] = {
                "dtype": dtype,
                "non_null_count": int(non_null_count),
                "pct_null": float(pct_null)
            }
        return schema


class DataProfilerTool:
    @staticmethod
    def profile(df: pd.DataFrame) -> Dict[str, Any]:
        profile = {}
        for col in df.columns:
            ser = df[col]
            profile[col] = {
                "n_unique": int(ser.nunique(dropna=True)),
                "pct_null": float(1 - ser.notna().sum() / max(1, len(ser)))
            }
            if pd.api.types.is_numeric_dtype(ser):
                clean = pd.to_numeric(ser, errors="coerce").dropna()
                if len(clean) > 0:
                    profile[col].update({
                        "mean": float(clean.mean()),
                        "std": float(clean.std()),
                        "min": float(clean.min()),
                        "max": float(clean.max())
                    })
        return profile


class OutlierDetectorTool:
    @staticmethod
    def detect_numeric_outliers(df: pd.DataFrame, cols: List[str]) -> Dict[str, List[int]]:
        outlier_idx = {}
        for col in cols:
            ser = pd.to_numeric(df[col], errors="coerce").dropna()
            if len(ser) < 10:
                outlier_idx[col] = []
                continue
            try:
                clf = IsolationForest(random_state=42, contamination="auto")
                preds = clf.fit_predict(ser.values.reshape(-1, 1))
                outliers = ser.index[preds == -1].tolist()
                outlier_idx[col] = outliers
            except Exception:
                outlier_idx[col] = []
        return outlier_idx


class DataImputerTool:
    @staticmethod
    def impute(df: pd.DataFrame,
               strategy_numeric="median",
               strategy_categorical="most_frequent") -> Tuple[pd.DataFrame, Dict[str, Any]]:
        df_out = df.copy()
        meta = {"imputations": {}}

        num_cols = df_out.select_dtypes(include=[np.number]).columns.tolist()
        if len(num_cols) > 0:
            imputer_num = SimpleImputer(strategy=strategy_numeric)
            df_out[num_cols] = imputer_num.fit_transform(df_out[num_cols])
            for c in num_cols:
                meta["imputations"][c] = strategy_numeric

        cat_cols = df_out.select_dtypes(exclude=[np.number]).columns.tolist()
        if len(cat_cols) > 0:
            imputer_cat = SimpleImputer(strategy=strategy_categorical, fill_value="__MISSING__")
            df_out[cat_cols] = imputer_cat.fit_transform(df_out[cat_cols])
            for c in cat_cols:
                meta["imputations"][c] = strategy_categorical

        return df_out, meta


class DuplicateResolverTool:
    @staticmethod
    def resolve(df: pd.DataFrame, subset: List[str] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        if subset is None:
            subset = df.columns.tolist()

        before = len(df)
        df_out = df.drop_duplicates(subset=subset)
        after = len(df_out)

        return df_out, {"removed_duplicates": before - after}


class DriftDetectorTool:
    @staticmethod
    def detect(baseline: pd.DataFrame,
               new: pd.DataFrame,
               cols: List[str] = None) -> Dict[str, Any]:
        drift = {}
        if cols is None:
            cols = baseline.columns.intersection(new.columns).tolist()
        for c in cols:
            a = pd.to_numeric(baseline[c], errors="coerce").dropna()
            b = pd.to_numeric(new[c], errors="coerce").dropna()
            if len(a) < 5 or len(b) < 5:
                drift[c] = {"status": "insufficient_data"}
                continue
            try:
                drift[c] = {
                    "mean_baseline": float(a.mean()),
                    "mean_new": float(b.mean()),
                    "mean_diff": float(b.mean() - a.mean())
                }
            except Exception:
                drift[c] = {"status": "error"}
        return drift


class FixGeneratorTool:
    @staticmethod
    def suggest(df: pd.DataFrame) -> Dict[str, Any]:
        suggestions = {}
        for c in df.columns:
            ser = df[c]
            if ser.isna().mean() > 0.2:
                suggestions[c] = "consider_drop_or_impute"
            elif ser.dtype == object and ser.nunique() > 1000:
                suggestions[c] = "high_cardinality"
            else:
                suggestions[c] = "clean_ok"
        return suggestions
