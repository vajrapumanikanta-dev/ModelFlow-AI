"""
Unit tests for Module 14: Report Generator (core/report.py)
"""

import os
from core.report import generate_html_report, generate_pdf_report


def test_html_and_pdf_report_generation(tmp_path):
    report_data = {
        "dataset_name": "test_dataset.csv",
        "problem_type": "binary_classification",
        "timestamp": "2026-10-04 12:00:00",
        "rows": 100,
        "cols": 5,
        "memory_usage": "15.5 KB",
        "missing_cells": 2,
        "missing_pct": 0.4,
        "duplicate_rows": 0,
        "target_col": "target",
        "cleaning_log": [{"timestamp": "12:00", "action": "impute", "columns": ["age"], "description": "imputed mean", "before_shape": [100, 5], "after_shape": [100, 5]}],
        "leakage_warnings": [],
        "imbalance_info": {"is_imbalanced": False, "reason": "Balanced"},
        "leaderboard_html": "<table class='table'><tr><th>Model</th><th>Accuracy</th></tr><tr><td>Random Forest</td><td>0.95</td></tr></table>",
        "leaderboard_table_data": [["Model", "Accuracy"], ["Random Forest", "0.95"]],
        "best_model_name": "Random Forest",
        "charts": [],
        "top_features": [{"feature": "age", "abs_importance": 0.65, "importance_pct": 65.0}],
    }

    # 1. HTML Report
    html_path = str(tmp_path / "report.html")
    html_out = generate_html_report(report_data, output_path=html_path)
    assert os.path.exists(html_path)
    assert "ModelFlow AI" in html_out
    assert "test_dataset.csv" in html_out

    # 2. PDF Report
    pdf_path = str(tmp_path / "report.pdf")
    pdf_out = generate_pdf_report(report_data, output_path=pdf_path)
    assert os.path.exists(pdf_path)
    assert os.path.getsize(pdf_path) > 1000
