# src/supervisor.py
from typing import Dict, Any
import pandas as pd
from .tools import SchemaInferTool, DataProfilerTool, FixGeneratorTool
from .agent import AdaptiveDataDoctorAgent

class SupervisorAgent:
    def __init__(self, baseline_path: str = None, outputs_dir: str = "outputs"):
        self.baseline_path = baseline_path
        self.outputs_dir = outputs_dir

    def run_full(self, path: str, evaluate_imputations: bool=False, target_column: str=None, problem_type: str="classification") -> Dict[str, Any]:
        # Stage 1: Schema + Profile
        df = pd.read_csv(path, low_memory=False)
        schema = SchemaInferTool.infer(df)
        profile = DataProfilerTool.profile(df)

        # Decide whether to call CleanerAgent (AdaptiveDataDoctorAgent)
        cleaner = AdaptiveDataDoctorAgent(baseline_path=self.baseline_path, outputs_dir=self.outputs_dir)
        # pass through evaluate_imputations if provided
        if evaluate_imputations and target_column:
            result = cleaner.run(path, evaluate_imputations=True, target_column=target_column, problem_type=problem_type)
        else:
            result = cleaner.run(path)
        # Stage 3: Post-clean suggestions
        cleaned = pd.read_csv(result["cleaned_path"], low_memory=False)
        suggestions = FixGeneratorTool.suggest(cleaned)

        return {
            "schema": schema,
            "profile": profile,
            "result": result,
            "suggestions": suggestions
        }
