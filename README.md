# google-x-kaggle
https://www.kaggle.com/code/shishirkatakam/notebook157133e39b

# AdaptiveDataDoctor

ADK-inspired agent that automatically detects and fixes common data quality issues.

## Features
- Schema inference & data profiling
- Missing-value imputation (auto + evaluative selector)
- Outlier detection (IsolationForest)
- Duplicate resolution
- Drift detection + visualization
- Audit report generation (Markdown)
- Optional Supervisor multi-agent orchestration

## Quick start
```bash
git clone https://github.com/shishir-katakam/google-x-kaggle
cd google-x-kaggle
pip install -r requirements.txt
python -c "from src.agent import AdaptiveDataDoctorAgent; AdaptiveDataDoctorAgent().run('data/sample_corrupted.csv')"
