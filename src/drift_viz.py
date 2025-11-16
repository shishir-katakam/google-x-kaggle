# src/drift_viz.py
import pandas as pd
import os
import matplotlib.pyplot as plt
from typing import List

def ensure_plot_dir(outputs_dir: str = "outputs"):
    plot_dir = os.path.join(outputs_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    return plot_dir

def plot_histogram_comparison(baseline: pd.Series, new: pd.Series, col: str, outpath: str):
    plt.figure()
    baseline_clean = pd.to_numeric(baseline, errors="coerce").dropna()
    new_clean = pd.to_numeric(new, errors="coerce").dropna()
    if baseline_clean.shape[0] > 0:
        plt.hist(baseline_clean, bins=30, alpha=0.5)
    if new_clean.shape[0] > 0:
        plt.hist(new_clean, bins=30, alpha=0.5)
    plt.title(f"Histogram: {col}")
    plt.xlabel(col)
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()

def plot_box_comparison(baseline: pd.Series, new: pd.Series, col: str, outpath: str):
    plt.figure()
    data = []
    labels = []
    baseline_clean = pd.to_numeric(baseline, errors="coerce").dropna()
    new_clean = pd.to_numeric(new, errors="coerce").dropna()
    if baseline_clean.shape[0] > 0:
        data.append(baseline_clean)
        labels.append("baseline")
    if new_clean.shape[0] > 0:
        data.append(new_clean)
        labels.append("new")
    if data:
        plt.boxplot(data, labels=labels)
        plt.title(f"Boxplot: {col}")
        plt.tight_layout()
        plt.savefig(outpath)
    plt.close()

def generate_drift_plots(baseline_df: pd.DataFrame, new_df: pd.DataFrame, cols: List[str]=None, outputs_dir: str="outputs"):
    plot_dir = ensure_plot_dir(outputs_dir)
    if cols is None:
        cols = baseline_df.columns.intersection(new_df.columns).tolist()
    saved = []
    for c in cols:
        try:
            hist_path = os.path.join(plot_dir, f"hist_{c}.png")
            box_path = os.path.join(plot_dir, f"box_{c}.png")
            plot_histogram_comparison(baseline_df[c], new_df[c], c, hist_path)
            plot_box_comparison(baseline_df[c], new_df[c], c, box_path)
            saved.append({"col": c, "hist": hist_path, "box": box_path})
        except Exception:
            continue
    return saved
