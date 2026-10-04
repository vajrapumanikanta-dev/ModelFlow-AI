"""
Module 14: Report Generation (core/report.py)
Generates comprehensive ML Analysis Reports in both HTML and PDF formats.
"""

import os
import io
import base64
from datetime import datetime
from typing import Dict, Any, List, Optional
import jinja2

# ReportLab imports for PDF generation
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image,
    KeepTogether,
    HRFlowable,
)


HTML_REPORT_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>ModelFlow AI - Machine Learning Analysis Report</title>
    <style>
        :root {
            --primary: #2563eb;
            --primary-light: #eff6ff;
            --text-main: #1f2937;
            --text-muted: #6b7280;
            --bg-card: #ffffff;
            --border: #e5e7eb;
            --success: #10b981;
        }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            color: var(--text-main);
            background-color: #f9fafb;
            margin: 0;
            padding: 40px 20px;
            line-height: 1.5;
        }
        .container {
            max-width: 1000px;
            margin: 0 auto;
        }
        .header {
            background: linear-gradient(135deg, #1e40af, #3b82f6);
            color: white;
            padding: 30px;
            border-radius: 12px;
            margin-bottom: 30px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        }
        .header h1 {
            margin: 0 0 10px 0;
            font-size: 28px;
            font-weight: 700;
        }
        .header .meta {
            font-size: 14px;
            opacity: 0.9;
        }
        .card {
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 24px;
            margin-bottom: 24px;
            box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05);
        }
        .card h2 {
            margin-top: 0;
            font-size: 18px;
            color: var(--primary);
            border-bottom: 2px solid var(--primary-light);
            padding-bottom: 8px;
        }
        .grid-2 {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 16px;
        }
        .grid-4 {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 12px;
        }
        .stat-box {
            background: #f8fafc;
            border: 1px solid var(--border);
            padding: 14px;
            border-radius: 8px;
            text-align: center;
        }
        .stat-box .num {
            font-size: 22px;
            font-weight: bold;
            color: var(--primary);
        }
        .stat-box .label {
            font-size: 12px;
            color: var(--text-muted);
            text-transform: uppercase;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 12px;
            font-size: 14px;
        }
        th, td {
            padding: 10px 12px;
            text-align: left;
            border-bottom: 1px solid var(--border);
        }
        th {
            background-color: #f3f4f6;
            font-weight: 600;
            color: #374151;
        }
        tr:hover {
            background-color: #f9fafb;
        }
        .badge {
            display: inline-block;
            padding: 3px 8px;
            border-radius: 9999px;
            font-size: 12px;
            font-weight: 600;
        }
        .badge-success { background: #d1fae5; color: #065f46; }
        .badge-warning { background: #fef3c7; color: #92400e; }
        .badge-info { background: #e0f2fe; color: #0369a1; }
        .chart-container {
            text-align: center;
            margin-top: 15px;
        }
        .chart-container img {
            max-width: 100%;
            height: auto;
            border-radius: 8px;
            border: 1px solid var(--border);
        }
        @media print {
            body { background: white; padding: 0; }
            .card { box-shadow: none; border: 1px solid #ddd; page-break-inside: avoid; }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>ModelFlow AI — ML Analysis Report</h1>
            <div class="meta">
                <span>Dataset: <strong>{{ dataset_name }}</strong></span> | 
                <span>Problem Type: <strong>{{ problem_type|capitalize }}</strong></span> | 
                <span>Generated: <strong>{{ timestamp }}</strong></span>
            </div>
        </div>

        <!-- 1. Dataset Overview -->
        <div class="card">
            <h2>1. Dataset Overview</h2>
            <div class="grid-4">
                <div class="stat-box">
                    <div class="num">{{ rows }}</div>
                    <div class="label">Total Rows</div>
                </div>
                <div class="stat-box">
                    <div class="num">{{ cols }}</div>
                    <div class="label">Total Columns</div>
                </div>
                <div class="stat-box">
                    <div class="num">{{ missing_cells }} ({{ missing_pct }}%)</div>
                    <div class="label">Missing Cells</div>
                </div>
                <div class="stat-box">
                    <div class="num">{{ duplicate_rows }}</div>
                    <div class="label">Duplicate Rows</div>
                </div>
            </div>
            <p style="margin-top: 15px; font-size: 14px;">
                <strong>Memory Usage:</strong> {{ memory_usage }} | 
                <strong>Target Column:</strong> <span class="badge badge-info">{{ target_col if target_col else 'None (Unsupervised)' }}</span>
            </p>
        </div>

        <!-- 2. Data Cleaning Log -->
        <div class="card">
            <h2>2. Data Cleaning History</h2>
            {% if cleaning_log and cleaning_log|length > 0 %}
            <table>
                <thead>
                    <tr>
                        <th>Time</th>
                        <th>Action</th>
                        <th>Columns</th>
                        <th>Description</th>
                        <th>Shape Change</th>
                    </tr>
                </thead>
                <tbody>
                    {% for step in cleaning_log %}
                    <tr>
                        <td>{{ step.timestamp }}</td>
                        <td><strong>{{ step.action }}</strong></td>
                        <td>{{ step.columns|join(', ') }}</td>
                        <td>{{ step.description }}</td>
                        <td>{{ step.before_shape }} &rarr; {{ step.after_shape }}</td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
            {% else %}
            <p style="color: var(--text-muted); font-size: 14px;">No cleaning modifications applied. Dataset processed in original state.</p>
            {% endif %}
        </div>

        <!-- 3. Quality & Leakage Checks -->
        <div class="card">
            <h2>3. Data Quality & Leakage Checks</h2>
            {% if leakage_warnings and leakage_warnings|length > 0 %}
            <table>
                <thead>
                    <tr>
                        <th>Column</th>
                        <th>Risk Level</th>
                        <th>Reason</th>
                        <th>Recommendation</th>
                    </tr>
                </thead>
                <tbody>
                    {% for w in leakage_warnings %}
                    <tr>
                        <td><strong>{{ w.column }}</strong></td>
                        <td><span class="badge badge-warning">{{ w.risk_level }}</span></td>
                        <td>{{ w.reason }}</td>
                        <td>{{ w.recommendation }}</td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
            {% else %}
            <p style="color: var(--success); font-size: 14px;">✓ No significant data leakage or high-risk proxy features detected.</p>
            {% endif %}
            {% if imbalance_info and imbalance_info.is_imbalanced %}
            <p style="margin-top: 10px; color: #b45309; font-size: 14px;">
                ⚠️ <strong>Class Imbalance Flagged:</strong> {{ imbalance_info.reason }}
            </p>
            {% endif %}
        </div>

        <!-- 4. Model Comparison & Leaderboard -->
        <div class="card">
            <h2>4. Model Performance Leaderboard</h2>
            {% if leaderboard_html %}
            {{ leaderboard_html|safe }}
            {% else %}
            <p>No model results available.</p>
            {% endif %}
            {% if best_model_name %}
            <div style="margin-top: 15px; padding: 12px; background: #ecfdf5; border-left: 4px solid #10b981; border-radius: 4px;">
                <strong>🏆 Recommended Best Model:</strong> {{ best_model_name }}
            </div>
            {% endif %}
        </div>

        <!-- 5. Key Visualizations -->
        {% if charts and charts|length > 0 %}
        <div class="card">
            <h2>5. Key Diagnostic Charts</h2>
            {% for chart in charts %}
            <div class="chart-container">
                <h3 style="font-size: 15px; margin-bottom: 6px;">{{ chart.title }}</h3>
                <img src="data:image/png;base64,{{ chart.base64 }}" alt="{{ chart.title }}">
            </div>
            <hr style="border: 0; border-top: 1px solid #eee; margin: 20px 0;">
            {% endfor %}
        </div>
        {% endif %}

        <!-- 6. Explainability Summary -->
        {% if top_features and top_features|length > 0 %}
        <div class="card">
            <h2>6. Feature Importance & Explainability</h2>
            <table>
                <thead>
                    <tr>
                        <th>Rank</th>
                        <th>Feature</th>
                        <th>Importance Score</th>
                        <th>Relative Contribution</th>
                    </tr>
                </thead>
                <tbody>
                    {% for f in top_features %}
                    <tr>
                        <td>#{{ loop.index }}</td>
                        <td><strong>{{ f.feature }}</strong></td>
                        <td>{{ "%.4f"|format(f.abs_importance) }}</td>
                        <td>{{ "%.2f"|format(f.importance_pct) }}%</td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
        </div>
        {% endif %}
    </div>
</body>
</html>
"""


def generate_html_report(report_data: Dict[str, Any], output_path: Optional[str] = None) -> str:
    """
    Renders styled HTML report from report data dictionary.
    """
    template = jinja2.Template(HTML_REPORT_TEMPLATE)
    html_content = template.render(**report_data)

    if output_path:
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)

    return html_content


def generate_pdf_report(report_data: Dict[str, Any], output_path: str = "reports/ml_report.pdf") -> str:
    """
    Generates downloadable PDF report using ReportLab.
    """
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36,
    )
    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        "TitleStyle",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#1e40af"),
        spaceAfter=6,
    )
    subtitle_style = ParagraphStyle(
        "SubTitleStyle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#4b5563"),
        spaceAfter=14,
    )
    h2_style = ParagraphStyle(
        "H2Style",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=17,
        textColor=colors.HexColor("#1e3a8a"),
        spaceBefore=12,
        spaceAfter=6,
    )
    body_style = ParagraphStyle(
        "BodyStyle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#1f2937"),
    )

    story = []

    # Title & Subtitle
    story.append(Paragraph("ModelFlow AI — Machine Learning Analysis Report", title_style))
    story.append(
        Paragraph(
            f"<b>Dataset:</b> {report_data.get('dataset_name', 'Dataset')} | "
            f"<b>Task:</b> {report_data.get('problem_type', 'ML').capitalize()} | "
            f"<b>Date:</b> {report_data.get('timestamp', datetime.now().strftime('%Y-%m-%d %H:%M'))}",
            subtitle_style,
        )
    )
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#2563eb"), spaceAfter=12))

    # 1. Dataset Overview Table
    story.append(Paragraph("1. Dataset Overview", h2_style))
    overview_data = [
        ["Total Rows", str(report_data.get("rows", "-")), "Total Columns", str(report_data.get("cols", "-"))],
        ["Missing Cells", f"{report_data.get('missing_cells', '-')} ({report_data.get('missing_pct', 0)}%)", "Duplicates", str(report_data.get("duplicate_rows", "-"))],
        ["Target Feature", str(report_data.get("target_col", "None")), "Memory Usage", str(report_data.get("memory_usage", "-"))],
    ]
    t_overview = Table(overview_data, colWidths=[120, 140, 120, 140])
    t_overview.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
            ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor("#1e293b")),
            ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
            ("FONTNAME", (2, 0), (2, -1), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8.5),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
            ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#cbd5e1")),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
        ])
    )
    story.append(t_overview)
    story.append(Spacer(1, 10))

    # 2. Cleaning Log
    story.append(Paragraph("2. Data Cleaning Log", h2_style))
    cleaning_log = report_data.get("cleaning_log", [])
    if cleaning_log:
        clean_rows = [["Time", "Action", "Columns", "Description"]]
        for c in cleaning_log[:8]:
            clean_rows.append([
                c.get("timestamp", "")[-8:],
                c.get("action", "")[:15],
                ", ".join(c.get("columns", []))[:20],
                c.get("description", "")[:45],
            ])
        t_clean = Table(clean_rows, colWidths=[70, 90, 110, 250])
        t_clean.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e2e8f0")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
                ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#cbd5e1")),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ])
        )
        story.append(t_clean)
    else:
        story.append(Paragraph("No interactive data cleaning was applied.", body_style))
    story.append(Spacer(1, 10))

    # 3. Leaderboard
    story.append(Paragraph("3. Model Performance Leaderboard", h2_style))
    leaderboard_table_data = report_data.get("leaderboard_table_data", [])
    if leaderboard_table_data and len(leaderboard_table_data) > 1:
        t_leader = Table(leaderboard_table_data, hAlign="LEFT")
        t_leader.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#3b82f6")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
                ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#cbd5e1")),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ])
        )
        story.append(t_leader)
    story.append(Spacer(1, 10))

    # 4. Embedded Charts
    charts = report_data.get("charts", [])
    if charts:
        story.append(Paragraph("4. Key Diagnostic Visualizations", h2_style))
        for chart in charts[:3]:  # Top 3 charts for PDF
            try:
                b64_str = chart.get("base64", "")
                if b64_str:
                    img_bytes = io.BytesIO(base64.b64decode(b64_str))
                    img = Image(img_bytes, width=420, height=240)
                    story.append(Paragraph(f"<b>{chart.get('title', '')}</b>", body_style))
                    story.append(Spacer(1, 4))
                    story.append(img)
                    story.append(Spacer(1, 10))
            except Exception:
                pass

    doc.build(story)
    return output_path
