# agent.py
import os
import pandas as pd

from .tools import (
    SchemaInferTool,
    DataProfilerTool,
    OutlierDetectorTool,
    DataImputerTool,
    DuplicateResolverTool,
    DriftDetectorTool,
    FixGeneratorTool
)
from .report_writer import write_report
from .imputation_tester import find_best_imputation
from .drift_viz import generate_drift_plots


class AdaptiveDataDoctorAgent:
    def __init__(self, baseline_path=None, outputs_dir="outputs",
                 evaluate_imputations=False, target_column=None,
                 problem_type="classification"):
        os.makedirs(outputs_dir, exist_ok=True)
        self.baseline_path = baseline_path
        self.outputs_dir = outputs_dir
        self.evaluate_imputations = evaluate_imputations
        self.target_column = target_column
        self.problem_type = problem_type

    def load(self, path):
        return pd.read_csv(path, low_memory=False)

    def run(self, path,
            evaluate_imputations=None,
            target_column=None,
            problem_type=None):

        # allow overrides when calling .run()
        if evaluate_imputations is None:
            evaluate_imputations = self.evaluate_imputations
        if target_column is None:
            target_column = self.target_column
        if problem_type is None:
            problem_type = self.problem_type

        df = self.load(path)
        print(f"Loaded {len(df)} rows, {len(df.columns)} columns")

        # ---------------- Schema & Profile ----------------
        schema = SchemaInferTool.infer(df)
        profile = DataProfilerTool.profile(df)

        # ---------------- Outlier Detection ----------------
        num_cols = df.select_dtypes(include=["number"]).columns.tolist()
        outliers = OutlierDetectorTool.detect_numeric_outliers(df, num_cols)

        # ---------------- Imputation (optional optimized) ----------------
        if evaluate_imputations and target_column and target_column in df.columns:
            print("\n🔎 Evaluating imputation strategies...")
            res = find_best_imputation(df, target=target_column, problem_type=problem_type)
            print("Imputation results:", res)

            if res.get("best"):
                num_strat = res["best"]["num_strategy"]
                cat_strat = res["best"]["cat_strategy"]
                print(f"Best strategies → numeric: {num_strat}, categorical: {cat_strat}")
                df_imputed, impute_meta = DataImputerTool.impute(
                    df,
                    strategy_numeric=num_strat,
                    strategy_categorical=cat_strat
                )
            else:
                df_imputed, impute_meta = DataImputerTool.impute(df)
        else:
            df_imputed, impute_meta = DataImputerTool.impute(df)

        # ---------------- Deduplication ----------------
        df_deduped, dedupe_meta = DuplicateResolverTool.resolve(df_imputed)

        # ---------------- Drift Detection + Plots ----------------
        drift = {}
        drift_plots = []

        if self.baseline_path:
            baseline = pd.read_csv(self.baseline_path, low_memory=False)
            drift = DriftDetectorTool.detect(baseline, df_deduped)
            try:
                drift_plots = generate_drift_plots(
                    baseline, df_deduped,
                    cols=None,
                    outputs_dir=self.outputs_dir
                )
            except Exception as e:
                print("⚠️ Drift plotting failed:", e)

        # ---------------- Final Suggestions ----------------
        suggestions = FixGeneratorTool.suggest(df_deduped)

        # ---------------- Save Cleaned Data ----------------
        cleaned_path = os.path.join(self.outputs_dir, "cleaned_output.csv")
        df_deduped.to_csv(cleaned_path, index=False)

        # ---------------- Save Report ----------------
        report_path = os.path.join(self.outputs_dir, "audit_report.md")
        write_report(
            filename=path,
            schema=schema,
            profile=profile,
            drift=drift,
            suggestions=suggestions,
            imputations=impute_meta.get("imputations", {}),
            drift_plots=drift_plots,
            out_path=report_path
        )

        print("\n✨ Cleaning complete!")
        print(f"Cleaned dataset → {cleaned_path}")
        print(f"Audit report → {report_path}")
        if drift_plots:
            print("Generated drift visualizations ✔")

        return {
            "cleaned_path": cleaned_path,
            "report_path": report_path,
            "drift_plots": drift_plots
        }
