
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


class AdaptiveDataDoctorAgent:
    def __init__(self, baseline_path=None, outputs_dir='outputs'):
        os.makedirs(outputs_dir, exist_ok=True)
        self.baseline_path = baseline_path
        self.outputs_dir = outputs_dir

    def load(self, path):
        return pd.read_csv(path, low_memory=False)

    def run(self, path):
        # Load
        df = self.load(path)
        print(f"Loaded {len(df)} rows, {len(df.columns)} columns")

        # Schema
        schema = SchemaInferTool.infer(df)

        # Profile
        profile = DataProfilerTool.profile(df)

        # Outliers (numeric only)
        num_cols = df.select_dtypes(include=['number']).columns.tolist()
        outliers = OutlierDetectorTool.detect_numeric_outliers(df, num_cols)

        # Imputation
        df_imputed, impute_meta = DataImputerTool.impute(df)

        # Deduplication
        df_deduped, dedupe_meta = DuplicateResolverTool.resolve(df_imputed)

        # Drift detection (if baseline exists)
        drift = {}
        if self.baseline_path:
            baseline = pd.read_csv(self.baseline_path, low_memory=False)
            drift = DriftDetectorTool.detect(baseline, df_deduped)

        # Fix suggestions
        suggestions = FixGeneratorTool.suggest(df_deduped)

        # Save cleaned data
        cleaned_path = os.path.join(self.outputs_dir, "cleaned_output.csv")
        df_deduped.to_csv(cleaned_path, index=False)

        # Save report
        report_path = os.path.join(self.outputs_dir, "audit_report.md")
        write_report(
            filename=path,
            schema=schema,
            profile=profile,
            drift=drift,
            suggestions=suggestions,
            imputations=impute_meta.get("imputations", {}),
            out_path=report_path
        )

        print(f"\nCleaned data → {cleaned_path}")
        print(f"Report → {report_path}")

        return {
            "cleaned_path": cleaned_path,
            "report_path": report_path
        }
