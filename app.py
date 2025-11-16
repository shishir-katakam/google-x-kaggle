# app.py
import streamlit as st
import tempfile
import os
import pandas as pd
from io import BytesIO

# ensure your src package is importable
# (in Streamlit cloud it will be, if repo root contains src/)
from src.agent import AdaptiveDataDoctorAgent
from src.supervisor import SupervisorAgent

st.set_page_config(page_title="AdaptiveDataDoctor", layout="wide")

st.title("AdaptiveDataDoctor — Automated Data Quality Agent")
st.markdown(
    """
Upload a CSV and the agent will:
- detect schema issues, missing values, outliers, duplicates
- optionally evaluate imputation strategies if you provide a labeled CSV or a target column
- show drift plots if you upload a baseline CSV
- return cleaned CSV, an audit report, and drift images
"""
)

# --- sidebar controls ---
st.sidebar.header("Run options")
use_supervisor = st.sidebar.checkbox("Run Supervisor pipeline (full)", value=True)
evaluate_imputations = st.sidebar.checkbox("Evaluate imputation strategies (requires labeled file / target)", value=False)
target_column = st.sidebar.text_input("Target column name (for imputation evaluation)", value="label")
use_baseline = st.sidebar.checkbox("Provide baseline CSV for drift detection", value=False)

uploaded_file = st.file_uploader("Upload dataset CSV (required)", type=["csv"])
baseline_file = st.file_uploader("Upload baseline CSV (optional, used if checked)", type=["csv"]) if use_baseline else None
# Optionally allow uploading a labeled dataset separate from main (if user wants)
labeled_file = st.file_uploader("Upload labeled CSV (optional) — will be used for imputation evaluation if provided", type=["csv"])

run_button = st.button("Run AdaptiveDataDoctor")

# utility to save uploaded file to disk for agent
def _save_uploaded(uploaded, dest_path):
    if uploaded is None:
        return None
    with open(dest_path, "wb") as f:
        f.write(uploaded.getvalue())
    return dest_path

def _make_downloadable_bytes(df: pd.DataFrame):
    b = BytesIO()
    df.to_csv(b, index=False)
    b.seek(0)
    return b

def _read_report(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"Could not read report: {e}"

# Run
if run_button:
    if uploaded_file is None:
        st.error("Please upload a dataset CSV to continue.")
    else:
        # workspace temp dir
        tmpdir = tempfile.mkdtemp(prefix="adoc_")
        os.makedirs(os.path.join(tmpdir, "outputs"), exist_ok=True)

        data_path = os.path.join(tmpdir, "input.csv")
        _save_uploaded(uploaded_file, data_path)

        baseline_path = None
        if baseline_file is not None:
            baseline_path = os.path.join(tmpdir, "baseline.csv")
            _save_uploaded(baseline_file, baseline_path)

        labeled_path = None
        if labeled_file is not None:
            labeled_path = os.path.join(tmpdir, "labeled.csv")
            _save_uploaded(labeled_file, labeled_path)

        st.info("Starting the agent — this may take a few seconds (or up to ~1–2 minutes on free tiers).")
        status = st.empty()
        try:
            # Choose pipeline
            if use_supervisor:
                status.text("Running SupervisorAgent (schema -> cleaner -> report)...")
                sup = SupervisorAgent(baseline_path=baseline_path, outputs_dir=os.path.join(tmpdir, "outputs"))
                # If user provided label and asked to evaluate, pass evaluate_imputations True
                if evaluate_imputations and (target_column or labeled_path):
                    # SupervisorAgent currently expects path + target; our run_full uses underlying cleaner.run to evaluate
                    res = sup.run_full(data_path, evaluate_imputations=evaluate_imputations, target_column=target_column)
                else:
                    res = sup.run_full(data_path, evaluate_imputations=False)
                # Supervisor returns a dict with "result"
                result = res.get("result", {})
                # If supervisor returned nothing, fallback to agent directly
                if not result:
                    status.text("Supervisor returned no result, falling back to direct agent run...")
                    agent = AdaptiveDataDoctorAgent(baseline_path=baseline_path, outputs_dir=os.path.join(tmpdir, "outputs"), evaluate_imputations=evaluate_imputations, target_column=target_column)
                    result = agent.run(data_path)
            else:
                status.text("Running direct agent...")
                agent = AdaptiveDataDoctorAgent(baseline_path=baseline_path, outputs_dir=os.path.join(tmpdir, "outputs"), evaluate_imputations=evaluate_imputations, target_column=target_column)
                result = agent.run(data_path)

            status.success("Agent finished successfully ✅")

            # Show cleaned dataset head
            cleaned_path = result.get("cleaned_path")
            report_path = result.get("report_path")
            drift_plots = result.get("drift_plots", [])

            st.subheader("Cleaned data (sample)")
            if cleaned_path and os.path.exists(cleaned_path):
                df_clean = pd.read_csv(cleaned_path)
                st.dataframe(df_clean.head(50))
                st.download_button("Download cleaned CSV", data=_make_downloadable_bytes(df_clean), file_name="cleaned_output.csv", mime="text/csv")
            else:
                st.warning("No cleaned CSV produced.")

            st.subheader("Audit report")
            if report_path and os.path.exists(report_path):
                txt = _read_report(report_path)
                st.code(txt, language="markdown")
                with open(report_path, "rb") as f:
                    rpt_bytes = f.read()
                st.download_button("Download audit report (MD)", data=rpt_bytes, file_name="audit_report.md", mime="text/markdown")
            else:
                st.warning("No audit report produced.")

            # Show drift plots
            if drift_plots:
                st.subheader("Drift visualizations")
                for item in drift_plots:
                    hist = item.get("hist")
                    box = item.get("box")
                    if hist and os.path.exists(hist):
                        st.image(hist, caption=f"{item.get('col')} - histogram")
                    if box and os.path.exists(box):
                        st.image(box, caption=f"{item.get('col')} - boxplot")

            # Log files list
            st.subheader("Outputs folder")
            files = []
            for root, dirs, filenames in os.walk(os.path.join(tmpdir, "outputs")):
                for fn in filenames:
                    files.append(os.path.join(root, fn))
            if files:
                st.write(files)
            else:
                st.write("No files generated in outputs/")

        except Exception as e:
            st.exception(e)
            status.error("Agent failed — check logs above.")
