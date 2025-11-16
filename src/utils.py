import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any
def read_csv_chunks(path, nrows=None, chunksize=None):
if chunksize:
return pd.read_csv(path, chunksize=chunksize, low_memory=False)
return pd.read_csv(path, nrows=nrows, low_memory=False)
def safe_cast_series(s: pd.Series):
try:
s_num = pd.to_numeric(s, errors='coerce')
if s_num.notna().sum() >= 0.8 * len(s):
return s_num
except Exception:
pass
return s
def sample_dataframe(df: pd.DataFrame, n=1000):
if len(df) <= n:
return df.copy()
return df.sample(n, random_state=42)