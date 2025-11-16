
from jinja2 import Template
import pandas as pd

REPORT_TMPL = """
# AdaptiveDataDoctor Audit Report

**Input file:** {{ filename }}

## Schema Summary
{% for col,info in schema.items() %}
- **{{ col }}**: dtype={{ info.dtype }}, pct_null={{ '%.2f'|format(info.pct_null) }}
{% endfor %}

## Profile Summary
{% for col,info in profile.items() %}
- **{{ col }}**: n_unique={{ info.n_unique }}, pct_null={{ '%.2f'|format(info.pct_null) }}
{% endfor %}

## Detected Drift
{% for col,info in drift.items() %}
- **{{ col }}**: {{ info }}
{% endfor %}

## Fix Suggestions
{% for col, s in suggestions.items() %}
- **{{ col }}**: {{ s }}
{% endfor %}

## Imputation Summary
{% for col, strat in imputations.items() %}
- **{{ col }}**: {{ strat }}
{% endfor %}

"""


def write_report(filename, schema, profile, drift, suggestions, imputations, out_path):
    tmpl = Template(REPORT_TMPL)
    txt = tmpl.render(filename=filename, schema=schema, profile=profile, drift=drift, suggestions=suggestions, imputations=imputations)
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(txt)
    return out_path