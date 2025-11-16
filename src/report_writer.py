# report_writer.py
from jinja2 import Template

REPORT_TMPL = """
# AdaptiveDataDoctor Audit Report

**Input file:** {{ filename }}

---

## 🧩 Schema Summary
{% for col,info in schema.items() %}
- **{{ col }}**  
  dtype = {{ info.dtype }}, null% = {{ "%.2f"|format(info.pct_null*100) }}%
{% endfor %}

---

## 📊 Profile Summary
{% for col,info in profile.items() %}
- **{{ col }}**  
  unique = {{ info.n_unique }}, null% = {{ "%.2f"|format(info.pct_null*100) }}%
  {% if info.mean %}
    (mean: {{ info.mean }}, std: {{ info.std }}, min: {{ info.min }}, max: {{ info.max }})
  {% endif %}
{% endfor %}

---

## 🔍 Drift Detection
{% if drift %}
{% for col,info in drift.items() %}
### {{ col }}
- {{ info }}
{% endfor %}
{% else %}
_No baseline provided — drift skipped._
{% endif %}

---

## 📉 Drift Plots
{% if drift_plots %}
{% for item in drift_plots %}
### {{ item.col }}
Histogram: {{ item.hist }}  
Boxplot: {{ item.box }}
{% endfor %}
{% else %}
_No drift plots available._
{% endif %}

---

## 🧼 Cleaning Suggestions
{% for col, s in suggestions.items() %}
- **{{ col }}**: {{ s }}
{% endfor %}

---

## 🔧 Imputation Summary
{% for col, strat in imputations.items() %}
- **{{ col }}** → {{ strat }}
{% endfor %}

---
Generated automatically by **AdaptiveDataDoctor**.
"""

def write_report(filename, schema, profile, drift, suggestions, imputations, drift_plots, out_path):
    tmpl = Template(REPORT_TMPL)
    txt = tmpl.render(
        filename=filename,
        schema=schema,
        profile=profile,
        drift=drift,
        suggestions=suggestions,
        imputations=imputations,
        drift_plots=drift_plots
    )
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(txt)
    return out_path
