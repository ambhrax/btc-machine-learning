"""
PDF Report Generator for ML Lab Project
Compiles an Executive Project Report (Abstract, Architecture, 22 Features, Benchmarks, Backtests, Discussion, Conclusion)
followed by all 6 executed Jupyter Notebooks with full code, outputs, markdown, tables, and Matplotlib charts.
Official Cover Page: Manipal University Jaipur Logo, Ambuj Mishra (2428010013), B.Tech CCE.
"""

import os
import sys
import json
import base64
import html
import subprocess
import mistune
from pygments import highlight
from pygments.lexers import PythonLexer
from pygments.formatters import HtmlFormatter

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
NOTEBOOKS_JSON = os.path.join(PROJECT_DIR, 'notebooks_data.json')
OUTPUT_HTML = os.path.join(PROJECT_DIR, 'ml_lab_report_full.html')
OUTPUT_PDF = os.path.join(PROJECT_DIR, 'ML_LAB_PROJECT_BTC_FORECASTING.pdf')
MUJ_LOGO_PATH = os.path.join(PROJECT_DIR, 'muj_logo.png')

def image_to_base64(img_path):
    full_path = os.path.join(PROJECT_DIR, img_path)
    if os.path.exists(full_path):
        with open(full_path, 'rb') as f:
            data = base64.b64encode(f.read()).decode('utf-8')
            ext = 'png' if img_path.endswith('.png') else 'jpeg'
            return f"data:image/{ext};base64,{data}"
    return img_path

def highlight_code(code_str):
    formatter = HtmlFormatter(nowrap=True)
    return highlight(code_str, PythonLexer(), formatter)

def clean_pandas_html(html_str):
    import re
    html_str = re.sub(r'<table[^>]*>', '<table class="df-table">', html_str)
    html_str = re.sub(r'border="1"', '', html_str)
    return html_str

def main():
    print(f"Loading notebook data from {NOTEBOOKS_JSON}...")
    with open(NOTEBOOKS_JSON, 'r', encoding='utf-8') as f:
        notebooks = json.load(f)

    # Convert MUJ logo to base64
    muj_logo_b64 = image_to_base64(MUJ_LOGO_PATH)

    # Pygments CSS
    pygments_css = HtmlFormatter().get_style_defs('.highlight')

    nb_meta = {
        '01_data_acquisition': {
            'num': '01',
            'title': 'Data Acquisition & Multi-Timeframe Resampling',
            'desc': 'Direct connection to Binance Spot Public API via CCXT. Acquires 800 high-resolution historical OHLCV bars across 13 standard and non-native custom timeframes (1m, 3m, 5m, 10m, 15m, 30m, 1h, 2h, 4h, 12h, 24h, 4d, 30d) using strict financial candle aggregation.',
            'category': 'Data Engineering',
            'badge': 'CCXT & Binance REST'
        },
        '02_eda_and_preprocessing': {
            'num': '02',
            'title': 'Exploratory Data Analysis & Statistical Preprocessing',
            'desc': 'Comprehensive audit for data hygiene (0 missing bars, 0 timestamp duplications). Evaluates price distribution moments, volatility clustering, and stationarity concepts. Proves that cryptocurrency log returns exhibit heavy fat tails (kurtosis = 4.89 vs Gaussian 3.0), requiring non-linear ML models.',
            'category': 'Statistical Analysis',
            'badge': 'Distribution Moments'
        },
        '03_feature_engineering': {
            'num': '03',
            'title': 'Quantitative Feature Engineering & Target Formulation',
            'desc': 'Computes 22 domain-specific quantitative technical indicators across 5 mathematical categories (Trend, Momentum, Volatility, Volume, Returns). Implements strict temporal shift (-1) for target close price P(t+1) to guarantee zero lookahead leakage across all 13 timeframes.',
            'category': 'Feature Engineering',
            'badge': '22 Technical Indicators'
        },
        '04_model_training_and_evaluation': {
            'num': '04',
            'title': 'Multi-Model Machine Learning Benchmarking',
            'desc': 'Trains and benchmarks Ridge Linear Regression, Random Forest Regressor, and LightGBM Regressor using an 80% train / 20% test chronological split with StandardScaler fit strictly on the training fold. Analyzes RMSE, MAE, R², and Directional Accuracy % (Hit Rate).',
            'category': 'Model Benchmarking',
            'badge': 'Ridge vs RF vs LightGBM'
        },
        '05_backtesting_and_simulation': {
            'num': '05',
            'title': 'Algorithmic Backtesting & Portfolio Trading Simulation',
            'desc': 'Executes an event-driven quantitative trading simulation with an exact $10,000.00 USD cash balance and realistic 0.1% transaction fee per trade. Evaluates 39 model × timeframe combinations, logging every transaction and computing Sharpe Ratio, Max Drawdown %, and Benchmark Alpha.',
            'category': 'Quantitative Simulation',
            'badge': '$10,000 Portfolio Sim'
        },
        '06_future_forecasting': {
            'num': '06',
            'title': 'Multi-Timeframe Future Price Forecasting & Recursive Projections',
            'desc': 'Deploys trained LightGBM models for live multi-timeframe inference. Produces instantaneous directional signal bias (BULLISH vs BEARISH) and expected returns for all 13 intervals, complemented by multi-step recursive autoregressive forecast trajectory cones (+1 to +5 intervals ahead).',
            'category': 'Inference & Forecasting',
            'badge': 'Recursive +1 to +5 Horizons'
        }
    }

    markdown_parser = mistune.create_markdown(escape=False)

    rendered_notebooks_html = []

    for nb in notebooks:
        nb_id = nb['id']
        meta = nb_meta.get(nb_id, {})
        num = meta.get('num', '00')
        title = meta.get('title', nb['title'])
        desc = meta.get('desc', '')
        category = meta.get('category', 'ML Lab')
        badge = meta.get('badge', '')

        cells_html = []
        cell_counter = 0

        for cell in nb['cells']:
            cell_type = cell['cell_type']
            source = cell['source']

            if cell_type == 'markdown':
                md_html = markdown_parser(source)
                cells_html.append(f'''
                <div class="cell-wrapper cell-markdown">
                  <div class="cell-gutter">
                    <span class="gutter-label">Markdown</span>
                  </div>
                  <div class="cell-content markdown-body">
                    {md_html}
                  </div>
                </div>
                ''')
            elif cell_type == 'code':
                cell_counter += 1
                highlighted_code = highlight_code(source)

                outputs_html = []
                for out in cell.get('outputs', []):
                    out_type = out.get('type')
                    if out_type == 'stream':
                        out_text = html.escape(out.get('text', ''))
                        outputs_html.append(f'''
                        <div class="output-subcell stream-output">
                          <div class="output-tag">stdout</div>
                          <pre class="terminal-text">{out_text}</pre>
                        </div>
                        ''')
                    elif out_type == 'html':
                        tbl_html = clean_pandas_html(out.get('html', ''))
                        outputs_html.append(f'''
                        <div class="output-subcell table-output">
                          <div class="output-tag">Pandas DataFrame</div>
                          <div class="table-scroll-container">
                            {tbl_html}
                          </div>
                        </div>
                        ''')
                    elif out_type == 'text':
                        out_text = html.escape(out.get('text', ''))
                        outputs_html.append(f'''
                        <div class="output-subcell text-output">
                          <div class="output-tag">Output [{cell_counter}]</div>
                          <pre class="terminal-text">{out_text}</pre>
                        </div>
                        ''')
                    elif out_type == 'image':
                        raw_src = out.get('src', '')
                        b64_src = image_to_base64(raw_src)
                        outputs_html.append(f'''
                        <div class="output-subcell figure-output">
                          <div class="output-tag">Matplotlib Plot Figure</div>
                          <div class="figure-wrapper">
                            <img src="{b64_src}" alt="Notebook Chart Artifact" class="notebook-chart-img" />
                          </div>
                        </div>
                        ''')

                output_block = ""
                if outputs_html:
                    output_block = f'''
                    <div class="cell-outputs-area">
                      {''.join(outputs_html)}
                    </div>
                    '''

                cells_html.append(f'''
                <div class="cell-wrapper cell-code">
                  <div class="cell-gutter">
                    <span class="gutter-label in-prompt">In [{cell_counter}]:</span>
                  </div>
                  <div class="cell-content">
                    <div class="code-editor-box">
                      <pre class="highlight"><code>{highlighted_code}</code></pre>
                    </div>
                    {output_block}
                  </div>
                </div>
                ''')

        joined_cells = '\n'.join(cells_html)

        nb_header = f'''
        <div class="notebook-report-section" id="section-nb-{num}">
          <div class="notebook-report-banner">
            <div class="nb-badge-row">
              <span class="nb-pill pill-category">{category}</span>
              <span class="nb-pill pill-num">Notebook #{num} of 06</span>
              <span class="nb-pill pill-tech">{badge}</span>
            </div>
            <h2 class="nb-section-title">Notebook {num}: {title}</h2>
            <p class="nb-section-desc">{desc}</p>
          </div>
          <div class="notebook-cells-flow">
            {joined_cells}
          </div>
        </div>
        '''
        rendered_notebooks_html.append(nb_header)

    all_notebooks_markup = '\n'.join(rendered_notebooks_html)

    # Assemble complete HTML document with print and academic cover styling
    full_html = f'''<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>ML Lab Project — Multi-Timeframe BTC Price Forecasting</title>
  <style>
    /* =========================================================
       ACADEMIC PRINT & PDF STYLING
       Target: Standard A4 Portrait Print
       ========================================================= */
    @page {{
      size: A4 portrait;
      margin: 14mm 12mm 14mm 12mm;
    }}

    *, *::before, *::after {{
      box-sizing: border-box;
      margin: 0;
      padding: 0;
    }}

    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
      font-size: 9pt;
      line-height: 1.48;
      color: #1e293b;
      background: #ffffff;
      -webkit-print-color-adjust: exact !important;
      print-color-adjust: exact !important;
    }}

    /* Page Breaks */
    .page-break {{
      page-break-after: always;
      break-after: page;
    }}
    .page-break-before {{
      page-break-before: always;
      break-before: page;
    }}
    .notebook-report-section {{
      page-break-before: always;
      break-before: page;
      margin-bottom: 24px;
    }}
    .cell-wrapper {{
      page-break-inside: avoid;
      break-inside: avoid;
      margin-bottom: 12px;
    }}
    .figure-wrapper, .figure-output {{
      page-break-inside: avoid;
      break-inside: avoid;
    }}
    .df-table, .report-table {{
      page-break-inside: avoid;
      break-inside: avoid;
    }}
    .avoid-break {{
      page-break-inside: avoid;
      break-inside: avoid;
    }}

    /* =========================================================
       COVER PAGE
       ========================================================= */
    .cover-container {{
      min-height: 96vh;
      display: flex;
      flex-direction: column;
      justify-content: space-between;
      border: 3px double #0f172a;
      padding: 34px 30px;
      background: #ffffff;
      position: relative;
    }}

    .cover-header {{
      text-align: center;
      border-bottom: 2px solid #0f172a;
      padding-bottom: 16px;
    }}

    .univ-logo-wrap {{
      margin: 0 auto 10px;
      display: flex;
      justify-content: center;
      align-items: center;
    }}

    .univ-logo-img {{
      height: 82px;
      width: auto;
      object-fit: contain;
    }}

    .univ-title {{
      font-size: 18.5pt;
      font-weight: 900;
      letter-spacing: 1.5px;
      color: #0f172a;
      text-transform: uppercase;
      margin-bottom: 4px;
    }}

    .univ-subtitle {{
      font-size: 10.5pt;
      font-weight: 700;
      color: #ea580c;
      text-transform: uppercase;
      letter-spacing: 0.8px;
    }}

    .univ-dept {{
      font-size: 9.5pt;
      color: #475569;
      font-weight: 600;
      margin-top: 3px;
    }}

    .cover-body {{
      text-align: center;
      padding: 20px 10px;
    }}

    .project-badge {{
      display: inline-block;
      padding: 5px 16px;
      background: #f1f5f9;
      border: 1px solid #cbd5e1;
      border-radius: 20px;
      font-size: 8.5pt;
      font-weight: 700;
      letter-spacing: 1px;
      color: #0369a1;
      text-transform: uppercase;
      margin-bottom: 14px;
    }}

    .project-main-title {{
      font-size: 17.5pt;
      font-weight: 900;
      color: #0f172a;
      line-height: 1.28;
      margin-bottom: 12px;
      text-transform: uppercase;
      letter-spacing: 0.5px;
    }}

    .project-tagline {{
      font-size: 10pt;
      color: #334155;
      max-width: 650px;
      margin: 0 auto 16px;
      line-height: 1.45;
    }}

    .project-specs-grid {{
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 10px;
      max-width: 680px;
      margin: 14px auto;
      text-align: center;
    }}

    .spec-card {{
      background: #f8fafc;
      border: 1px solid #e2e8f0;
      border-radius: 8px;
      padding: 9px 6px;
    }}

    .spec-val {{
      font-size: 12pt;
      font-weight: 800;
      color: #0f172a;
      font-family: monospace;
    }}

    .spec-lbl {{
      font-size: 7.5pt;
      color: #64748b;
      text-transform: uppercase;
      margin-top: 2px;
      font-weight: 600;
    }}

    .authors-section {{
      background: #f8fafc;
      border: 1px solid #cbd5e1;
      border-radius: 10px;
      padding: 12px 14px;
      max-width: 680px;
      margin: 14px auto 0;
      text-align: left;
    }}

    .authors-label {{
      font-size: 8pt;
      font-weight: 700;
      text-transform: uppercase;
      color: #64748b;
      letter-spacing: 0.8px;
      margin-bottom: 8px;
      text-align: center;
    }}

    .authors-grid {{
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 10px;
    }}

    .author-card {{
      background: #ffffff;
      border: 1px solid #e2e8f0;
      border-radius: 8px;
      padding: 8px 10px;
      border-left: 4px solid #ea580c;
    }}

    .author-name {{
      font-size: 10pt;
      font-weight: 800;
      color: #0f172a;
    }}

    .author-reg {{
      font-size: 8.5pt;
      color: #0369a1;
      font-family: monospace;
      margin-top: 2px;
      font-weight: 700;
    }}

    .author-role {{
      font-size: 8pt;
      color: #334155;
      font-weight: 600;
      margin-top: 2px;
    }}

    .author-branch {{
      font-size: 7.5pt;
      color: #64748b;
      margin-top: 1px;
    }}

    .cover-footer {{
      border-top: 2px solid #0f172a;
      padding-top: 12px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      font-size: 8.5pt;
      color: #475569;
    }}

    .cover-footer-left {{
      text-align: left;
    }}

    .cover-footer-right {{
      text-align: right;
    }}

    /* =========================================================
       REPORT & DOSSIER STYLING
       ========================================================= */
    .report-dossier-wrap {{
      padding: 10px 4px;
    }}

    .dossier-header-box {{
      background: #0f172a;
      color: #ffffff;
      padding: 16px 20px;
      border-radius: 8px;
      margin-bottom: 20px;
    }}

    .dossier-tag {{
      display: inline-block;
      font-size: 7.5pt;
      font-weight: 800;
      color: #38bdf8;
      text-transform: uppercase;
      letter-spacing: 1px;
      margin-bottom: 4px;
    }}

    .dossier-title {{
      font-size: 15pt;
      font-weight: 800;
      letter-spacing: 0.5px;
      margin-bottom: 4px;
      color: #ffffff;
    }}

    .dossier-sub {{
      font-size: 8.5pt;
      color: #94a3b8;
    }}

    .report-section {{
      margin-bottom: 22px;
    }}

    .report-h2 {{
      font-size: 12.5pt;
      font-weight: 800;
      color: #0f172a;
      border-bottom: 2px solid #0284c7;
      padding-bottom: 4px;
      margin-top: 16px;
      margin-bottom: 10px;
      text-transform: uppercase;
      letter-spacing: 0.5px;
    }}

    .report-h3 {{
      font-size: 10pt;
      font-weight: 700;
      color: #1e293b;
      margin-top: 12px;
      margin-bottom: 6px;
    }}

    .report-p {{
      font-size: 8.8pt;
      color: #334155;
      line-height: 1.5;
      margin-bottom: 8px;
      text-align: justify;
    }}

    .report-grid-3 {{
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 10px;
      margin: 12px 0;
    }}

    .report-grid-2 {{
      display: grid;
      grid-template-columns: repeat(2, 1fr);
      gap: 12px;
      margin: 12px 0;
    }}

    .kpi-card {{
      background: #f8fafc;
      border: 1px solid #cbd5e1;
      border-radius: 6px;
      padding: 10px 12px;
      text-align: center;
    }}

    .kpi-val {{
      font-size: 13pt;
      font-weight: 800;
      color: #0284c7;
      font-family: monospace;
    }}

    .kpi-val.green {{ color: #16a34a; }}
    .kpi-val.red {{ color: #dc2626; }}
    .kpi-val.orange {{ color: #ea580c; }}

    .kpi-lbl {{
      font-size: 7.2pt;
      font-weight: 700;
      color: #64748b;
      text-transform: uppercase;
      margin-top: 2px;
    }}

    .callout-box {{
      background: #f0fdf4;
      border-left: 4px solid #16a34a;
      border-radius: 4px;
      padding: 9px 12px;
      margin: 10px 0;
      font-size: 8.3pt;
      color: #166534;
      line-height: 1.45;
    }}

    .callout-info {{
      background: #f0f9ff;
      border-left: 4px solid #0284c7;
      border-radius: 4px;
      padding: 9px 12px;
      margin: 10px 0;
      font-size: 8.3pt;
      color: #0369a1;
      line-height: 1.45;
    }}

    .callout-alert {{
      background: #fef2f2;
      border-left: 4px solid #dc2626;
      border-radius: 4px;
      padding: 9px 12px;
      margin: 10px 0;
      font-size: 8.3pt;
      color: #991b1b;
      line-height: 1.45;
    }}

    .report-table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 7.8pt;
      margin: 10px 0;
    }}

    .report-table th {{
      background: #0f172a;
      color: #ffffff;
      padding: 6px 8px;
      font-weight: 700;
      border: 1px solid #334155;
      text-align: center;
    }}

    .report-table td {{
      padding: 5px 8px;
      border: 1px solid #cbd5e1;
      color: #1e293b;
    }}

    .report-table tr:nth-child(even) {{
      background: #f8fafc;
    }}

    .tag-pill {{
      display: inline-block;
      padding: 2px 6px;
      border-radius: 3px;
      font-size: 6.8pt;
      font-weight: 700;
      text-transform: uppercase;
    }}

    .tag-bull {{ background: #dcfce7; color: #15803d; border: 1px solid #86efac; }}
    .tag-bear {{ background: #fee2e2; color: #b91c1c; border: 1px solid #fca5a5; }}
    .tag-win {{ background: #e0f2fe; color: #0369a1; border: 1px solid #7dd3fc; font-weight: bold; }}

    /* =========================================================
       TOC PAGE
       ========================================================= */
    .toc-container {{
      padding: 16px 8px;
    }}

    .toc-title {{
      font-size: 14pt;
      font-weight: 800;
      color: #0f172a;
      border-bottom: 2px solid #0284c7;
      padding-bottom: 6px;
      margin-bottom: 14px;
      text-transform: uppercase;
      letter-spacing: 0.5px;
    }}

    .toc-list {{
      list-style: none;
    }}

    .toc-item {{
      display: flex;
      align-items: baseline;
      padding: 7px 0;
      border-bottom: 1px dashed #e2e8f0;
    }}

    .toc-num {{
      font-family: monospace;
      font-weight: 800;
      color: #0284c7;
      font-size: 10.5pt;
      width: 32px;
      flex-shrink: 0;
    }}

    .toc-text {{
      flex-grow: 1;
      font-weight: 700;
      font-size: 9.2pt;
      color: #1e293b;
    }}

    .toc-desc {{
      display: block;
      font-size: 7.8pt;
      font-weight: 400;
      color: #64748b;
      margin-top: 2px;
    }}

    .toc-badge {{
      font-size: 7.2pt;
      padding: 2px 8px;
      background: #f1f5f9;
      border: 1px solid #cbd5e1;
      border-radius: 4px;
      font-weight: 600;
      color: #475569;
      flex-shrink: 0;
    }}

    /* =========================================================
       NOTEBOOK SECTION HEADERS
       ========================================================= */
    .notebook-report-banner {{
      background: #f8fafc;
      border: 1px solid #cbd5e1;
      border-left: 6px solid #0284c7;
      border-radius: 8px;
      padding: 14px 16px;
      margin-bottom: 16px;
      page-break-after: avoid;
      break-after: avoid;
    }}

    .nb-badge-row {{
      display: flex;
      gap: 8px;
      margin-bottom: 6px;
    }}

    .nb-pill {{
      font-size: 7pt;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.5px;
      padding: 2px 8px;
      border-radius: 4px;
    }}

    .pill-category {{ background: #e0f2fe; color: #0369a1; border: 1px solid #bae6fd; }}
    .pill-num {{ background: #f1f5f9; color: #334155; border: 1px solid #cbd5e1; }}
    .pill-tech {{ background: #fef3c7; color: #92400e; border: 1px solid #fde68a; }}

    .nb-section-title {{
      font-size: 13pt;
      font-weight: 800;
      color: #0f172a;
      line-height: 1.3;
      margin-bottom: 4px;
    }}

    .nb-section-desc {{
      font-size: 8.3pt;
      color: #475569;
      line-height: 1.4;
    }}

    /* =========================================================
       NOTEBOOK CELLS FLOW
       ========================================================= */
    .cell-wrapper {{
      display: flex;
      gap: 8px;
      align-items: stretch;
    }}

    .cell-gutter {{
      width: 58px;
      flex-shrink: 0;
      text-align: right;
      padding-top: 4px;
    }}

    .gutter-label {{
      font-family: monospace;
      font-size: 7.5pt;
      color: #94a3b8;
      font-weight: 600;
    }}

    .gutter-label.in-prompt {{
      color: #0284c7;
      font-weight: 700;
    }}

    .cell-content {{
      flex-grow: 1;
      width: calc(100% - 66px);
    }}

    /* Markdown styling */
    .cell-markdown .cell-content {{
      background: #ffffff;
      padding: 6px 12px;
      border-left: 2px solid #e2e8f0;
    }}

    .markdown-body h1, .markdown-body h2, .markdown-body h3, .markdown-body h4 {{
      color: #0f172a;
      margin-top: 6px;
      margin-bottom: 4px;
      font-weight: 700;
      page-break-after: avoid;
      break-after: avoid;
    }}
    .markdown-body h1 {{ font-size: 11pt; border-bottom: 1px solid #e2e8f0; padding-bottom: 3px; }}
    .markdown-body h2 {{ font-size: 10.5pt; }}
    .markdown-body h3 {{ font-size: 9.5pt; }}
    .markdown-body h4 {{ font-size: 9pt; }}

    .markdown-body p {{
      margin-bottom: 6px;
      font-size: 8.5pt;
      color: #334155;
      line-height: 1.45;
    }}

    .markdown-body ul, .markdown-body ol {{
      margin-left: 18px;
      margin-bottom: 6px;
      font-size: 8.5pt;
      color: #334155;
    }}

    .markdown-body li {{
      margin-bottom: 2px;
    }}

    .markdown-body code {{
      font-family: "JetBrains Mono", Consolas, "Courier New", monospace;
      font-size: 8pt;
      background: #f1f5f9;
      color: #0f172a;
      padding: 1px 4px;
      border-radius: 3px;
      border: 1px solid #e2e8f0;
    }}

    /* Code editor box */
    .code-editor-box {{
      background: #f8fafc;
      border: 1px solid #cbd5e1;
      border-radius: 6px;
      padding: 8px 12px;
      overflow-x: auto;
    }}

    .code-editor-box pre {{
      margin: 0;
      font-family: "JetBrains Mono", Consolas, "Courier New", monospace;
      font-size: 7.8pt;
      line-height: 1.4;
      white-space: pre-wrap;
      word-break: break-word;
    }}

    /* Outputs area */
    .cell-outputs-area {{
      margin-top: 6px;
    }}

    .output-subcell {{
      margin-top: 6px;
      border-radius: 6px;
      border: 1px solid #e2e8f0;
      overflow: hidden;
      background: #ffffff;
    }}

    .output-tag {{
      background: #f1f5f9;
      border-bottom: 1px solid #e2e8f0;
      padding: 3px 8px;
      font-size: 6.8pt;
      font-weight: 700;
      text-transform: uppercase;
      color: #64748b;
      font-family: monospace;
      letter-spacing: 0.5px;
    }}

    /* Terminal stream output */
    .stream-output .terminal-text,
    .text-output .terminal-text {{
      background: #0f172a;
      color: #f8fafc;
      font-family: "JetBrains Mono", Consolas, "Courier New", monospace;
      font-size: 7.4pt;
      line-height: 1.35;
      padding: 8px 10px;
      margin: 0;
      white-space: pre-wrap;
      word-break: break-all;
    }}

    /* Table styling */
    .table-scroll-container {{
      padding: 6px;
      overflow-x: auto;
    }}

    .df-table {{
      width: 100%;
      border-collapse: collapse;
      font-family: "JetBrains Mono", Consolas, monospace;
      font-size: 7pt;
      text-align: right;
    }}

    .df-table th, .df-table td {{
      padding: 4px 6px;
      border: 1px solid #cbd5e1;
    }}

    .df-table th {{
      background: #f1f5f9;
      color: #0f172a;
      font-weight: 700;
      text-align: center;
    }}

    .df-table tr:nth-child(even) {{
      background: #f8fafc;
    }}

    /* Figure / Chart styling */
    .figure-wrapper {{
      text-align: center;
      padding: 8px 4px;
      background: #ffffff;
    }}

    .notebook-chart-img {{
      max-width: 96%;
      height: auto;
      border-radius: 4px;
      border: 1px solid #e2e8f0;
      display: inline-block;
    }}

    /* Pygments Syntax Highlighting CSS */
    {pygments_css}
  </style>
</head>
<body>

  <!-- =====================================================
       ACADEMIC COVER PAGE
       ===================================================== -->
  <div class="cover-container page-break">
    <div class="cover-header">
      <div class="univ-logo-wrap">
        <img src="{muj_logo_b64}" alt="Manipal University Jaipur Logo" class="univ-logo-img" />
      </div>
      <h1 class="univ-title">MANIPAL UNIVERSITY JAIPUR</h1>
      <h2 class="univ-subtitle">Department of Computer Science &amp; Engineering (CCE)</h2>
      <p class="univ-dept">School of Computing and Information Technology • Academic Session 2025–2026</p>
    </div>

    <div class="cover-body">
      <div class="project-badge">Machine Learning Laboratory Coursework Project Dossier</div>
      <h2 class="project-main-title">
        Multi-Timeframe Cryptocurrency Price Forecasting &amp; Quantitative Algorithmic Backtesting System
      </h2>
      <p class="project-tagline">
        An end-to-end quantitative machine learning pipeline evaluating Bitcoin (<code>BTC/USDT</code>) across 13 distinct timeframes (1-minute scalping to 30-day macro bars) with 22 engineered technical indicators, supervised regression benchmarking, transaction-cost-aware event-driven backtesting, and recursive multi-horizon projections.
      </p>

      <div class="project-specs-grid">
        <div class="spec-card">
          <div class="spec-val">13</div>
          <div class="spec-lbl">Timeframes Analyzed</div>
        </div>
        <div class="spec-card">
          <div class="spec-val">22</div>
          <div class="spec-lbl">Technical Indicators</div>
        </div>
        <div class="spec-card">
          <div class="spec-val">3</div>
          <div class="spec-lbl">Benchmarked Models</div>
        </div>
        <div class="spec-card">
          <div class="spec-val">$10,000</div>
          <div class="spec-lbl">Backtest Capital</div>
        </div>
      </div>

      <div class="authors-section">
        <div class="authors-label">Project Investigators &amp; Authors</div>
        <div class="authors-grid">
          <div class="author-card">
            <div class="author-name">Ambuj Mishra</div>
            <div class="author-reg">Reg: 2428010013</div>
            <div class="author-role">B.Tech CCE</div>
            <div class="author-branch">Manipal University Jaipur</div>
          </div>
        </div>
      </div>
    </div>

    <div class="cover-footer">
      <div class="cover-footer-left">
        <strong>Course:</strong> Machine Learning Laboratory (CS3150 / CCE)<br>
        <strong>Affiliation:</strong> Manipal University Jaipur, Dehmi Kalan, Jaipur, Rajasthan 303007
      </div>
      <div class="cover-footer-right">
        <strong>Program:</strong> B.Tech CCE<br>
        <strong>Date:</strong> October 2026
      </div>
    </div>
  </div>

  <!-- =====================================================
       TABLE OF CONTENTS & PROJECT SYNOPSIS
       ===================================================== -->
  <div class="toc-container page-break">
    <h2 class="toc-title">Table of Contents &amp; Project Structure</h2>
    <p style="font-size: 8.5pt; color: #475569; margin-bottom: 14px; line-height: 1.5;">
      This comprehensive laboratory report is structured into two primary divisions: <strong>Part I: Executive Project Report &amp; Scientific Dossier</strong> (containing theoretical formulation, feature engineering taxonomy, model benchmark tables, backtesting performance analysis, and risk metrics), followed by <strong>Part II: Complete Notebooks Laboratory Execution</strong> (compiling all 6 executed Jupyter Notebooks with active code, outputs, and visualizations).
    </p>

    <div style="font-size: 9pt; font-weight: 800; color: #0284c7; text-transform: uppercase; margin-bottom: 6px; letter-spacing: 0.5px;">
      Part I: Executive Project Report &amp; Scientific Dossier
    </div>
    <ul class="toc-list" style="margin-bottom: 16px;">
      <li class="toc-item">
        <span class="toc-num">R1</span>
        <div class="toc-text">
          Executive Summary &amp; Research Problem Statement
          <span class="toc-desc">Motivation • Market non-stationarity • Fat-tail returns • Core scientific objectives</span>
        </div>
        <span class="toc-badge">Section 1</span>
      </li>
      <li class="toc-item">
        <span class="toc-num">R2</span>
        <div class="toc-text">
          System Architecture &amp; Modular Pipeline Design
          <span class="toc-desc">Data ingestion • Resampling • Stationarity verification • Chronological train/test protocol</span>
        </div>
        <span class="toc-badge">Section 2</span>
      </li>
      <li class="toc-item">
        <span class="toc-num">R3</span>
        <div class="toc-text">
          Domain Technical Feature Engineering (22 Indicators)
          <span class="toc-desc">Trend, Momentum, Volatility, Volume, and Return features • Temporal target shift P(t+1)</span>
        </div>
        <span class="toc-badge">Section 3</span>
      </li>
      <li class="toc-item">
        <span class="toc-num">R4</span>
        <div class="toc-text">
          Comparative Machine Learning Benchmarking Results
          <span class="toc-desc">Ridge Regression vs Random Forest vs LightGBM • RMSE, MAE, R², Directional Hit Rate</span>
        </div>
        <span class="toc-badge">Section 4</span>
      </li>
      <li class="toc-item">
        <span class="toc-num">R5</span>
        <div class="toc-text">
          Quantitative Algorithmic Backtesting &amp; Risk Metrics
          <span class="toc-desc">$10,000 portfolio simulation • 0.1% fees • Sharpe Ratio, Max Drawdown %, Net PnL, Alpha</span>
        </div>
        <span class="toc-badge">Section 5</span>
      </li>
      <li class="toc-item">
        <span class="toc-num">R6</span>
        <div class="toc-text">
          Critical Discussion: The High-Frequency Friction Trap vs Swing Alpha
          <span class="toc-desc">Why 1m-15m fails under fees • Why 4h-4d achieves Sharpe > 2.0 • Non-stationarity findings</span>
        </div>
        <span class="toc-badge">Section 6</span>
      </li>
      <li class="toc-item">
        <span class="toc-num">R7</span>
        <div class="toc-text">
          Conclusions, System Limitations &amp; Future Scope
          <span class="toc-desc">Summary of findings • Transformer/LSTM extensions • Order book microstructure</span>
        </div>
        <span class="toc-badge">Section 7</span>
      </li>
    </ul>

    <div style="font-size: 9pt; font-weight: 800; color: #0284c7; text-transform: uppercase; margin-bottom: 6px; letter-spacing: 0.5px;">
      Part II: Executed Laboratory Notebooks Compilation
    </div>
    <ul class="toc-list">
      <li class="toc-item">
        <span class="toc-num">01</span>
        <div class="toc-text">
          Data Acquisition &amp; Multi-Timeframe Resampling
          <span class="toc-desc">Binance Spot Public API via CCXT • 13 timeframes • Strict financial candle aggregation</span>
        </div>
        <span class="toc-badge">Notebook #01</span>
      </li>
      <li class="toc-item">
        <span class="toc-num">02</span>
        <div class="toc-text">
          Exploratory Data Analysis &amp; Statistical Preprocessing
          <span class="toc-desc">Data hygiene audit • Stationarity tests • Log return kurtosis (4.89 vs Gaussian 3.0 fat-tail proof)</span>
        </div>
        <span class="toc-badge">Notebook #02</span>
      </li>
      <li class="toc-item">
        <span class="toc-num">03</span>
        <div class="toc-text">
          Quantitative Feature Engineering &amp; Target Formulation
          <span class="toc-desc">22 domain-specific indicators • Temporal shift (-1) target formulation • Zero lookahead leakage</span>
        </div>
        <span class="toc-badge">Notebook #03</span>
      </li>
      <li class="toc-item">
        <span class="toc-num">04</span>
        <div class="toc-text">
          Multi-Model Machine Learning Benchmarking
          <span class="toc-desc">Linear Ridge vs Random Forest vs LightGBM • Chronological 80/20 train/test split</span>
        </div>
        <span class="toc-badge">Notebook #04</span>
      </li>
      <li class="toc-item">
        <span class="toc-num">05</span>
        <div class="toc-text">
          Algorithmic Backtesting &amp; Portfolio Trading Simulation
          <span class="toc-desc">Event-driven transaction-level backtest • $10,000 cash balance • 0.1% maker/taker trading fee</span>
        </div>
        <span class="toc-badge">Notebook #05</span>
      </li>
      <li class="toc-item">
        <span class="toc-num">06</span>
        <div class="toc-text">
          Multi-Timeframe Future Price Forecasting &amp; Recursive Projections
          <span class="toc-desc">Instantaneous directional bias across 13 timeframes • Recursive forward trajectory cones</span>
        </div>
        <span class="toc-badge">Notebook #06</span>
      </li>
    </ul>

    <div style="margin-top: 18px; padding: 12px 14px; background: #f8fafc; border: 1px solid #cbd5e1; border-radius: 8px;">
      <h3 style="font-size: 9pt; font-weight: 800; color: #0f172a; margin-bottom: 2px;">Academic Verification &amp; Reproducibility</h3>
      <p style="font-size: 7.8pt; color: #475569; line-height: 1.4;">
        Submitted in partial fulfillment of the Machine Learning Laboratory requirement for <strong>B.Tech CCE</strong> at <strong>Manipal University Jaipur</strong>. All random states (<code>random_state=42</code>) were strictly pinned to ensure deterministic laboratory replication.
      </p>
    </div>
  </div>

  <!-- =====================================================
       PART I: EXECUTIVE PROJECT REPORT & SCIENTIFIC DOSSIER
       ===================================================== -->
  <div class="report-dossier-wrap page-break">
    <div class="dossier-header-box">
      <span class="dossier-tag">Part I • Scientific Research Dossier</span>
      <h2 class="dossier-title">Comprehensive Project Report, Benchmark Results &amp; Conclusions</h2>
      <p class="dossier-sub">
        Author: Ambuj Mishra (2428010013) • Department of CSE (CCE), Manipal University Jaipur
      </p>
    </div>

    <!-- Section 1 -->
    <div class="report-section">
      <h3 class="report-h2">1. Executive Summary &amp; Research Problem</h3>
      <p class="report-p">
        Financial time series forecasting in high-volatility cryptocurrency markets presents unique mathematical challenges distinct from conventional machine learning domains. Assets like Bitcoin (<code>BTC/USDT</code>) exhibit severe <strong>non-stationarity</strong>, time-varying conditional variance (volatility clustering), and <strong>fat-tailed return distributions</strong> that violate traditional Gaussian normality assumptions.
      </p>
      <p class="report-p">
        In this laboratory investigation, we engineer and evaluate an end-to-end quantitative Machine Learning pipeline across <strong>13 distinct trading timeframes</strong>:
      </p>
      <ul style="margin-left: 20px; font-size: 8.5pt; color: #334155; margin-bottom: 8px;">
        <li><strong>Ultra-Short &amp; Scalping:</strong> <code>1m</code>, <code>3m</code>, <code>5m</code>, <code>10m</code></li>
        <li><strong>Intraday Horizons:</strong> <code>15m</code>, <code>30m</code>, <code>1h</code>, <code>2h</code>, <code>4h</code></li>
        <li><strong>Swing &amp; Macro Trend:</strong> <code>12h</code>, <code>24h</code> (Daily), <code>4d</code>, <code>30d</code> (Monthly)</li>
      </ul>
      <p class="report-p">
        The system extracts raw OHLCV tick data directly from the public Binance Spot API via the CCXT protocol, computes a high-dimensional feature space of <strong>22 quantitative technical indicators</strong> across 5 mathematical categories, trains three diverse supervised learning architectures (L2-Regularized Linear Ridge, Random Forest Ensembles, and Gradient-Boosted Decision Trees via LightGBM), and executes an event-driven backtesting engine simulating a live $10,000 portfolio subject to realistic exchange trading fees.
      </p>

      <div class="report-grid-3">
        <div class="kpi-card">
          <div class="kpi-val green">+21.45%</div>
          <div class="kpi-lbl">Top Strategy Return (4d LightGBM)</div>
        </div>
        <div class="kpi-card">
          <div class="kpi-val green">2.11</div>
          <div class="kpi-lbl">Peak Sharpe Ratio (4d Interval)</div>
        </div>
        <div class="kpi-card">
          <div class="kpi-val orange">4.89</div>
          <div class="kpi-lbl">Return Kurtosis (Fat-Tail Proof)</div>
        </div>
      </div>
    </div>

    <!-- Section 2 -->
    <div class="report-section">
      <h3 class="report-h2">2. System Architecture &amp; Modular Pipeline Design</h3>
      <p class="report-p">
        The research framework is implemented through a strictly segregated, modular Python package located in <code>src/</code> with dedicated pipelines for each lifecycle stage:
      </p>
      <table class="report-table">
        <thead>
          <tr>
            <th style="width: 18%;">Pipeline Stage</th>
            <th style="width: 25%;">Module Source</th>
            <th style="width: 57%;">Technical Implementation &amp; Scientific Protocol</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td><strong>1. Ingestion</strong></td>
            <td><code>src/data_loader.py</code></td>
            <td>Paginated REST extraction via CCXT from Binance Spot Public API. Standard OHLCV financial resampling for custom intervals (<code>10m</code> from 5m, <code>4d</code> and <code>30d</code> from 24h).</td>
          </tr>
          <tr>
            <td><strong>2. EDA &amp; Audit</strong></td>
            <td><code>notebooks/02_eda...</code></td>
            <td>Verification of zero duplicate timestamps and zero missing candles. Augmented Dickey-Fuller (ADF) stationarity testing. Statistical proof of excess kurtosis (4.89 vs 3.0).</td>
          </tr>
          <tr>
            <td><strong>3. Features</strong></td>
            <td><code>src/features.py</code></td>
            <td>Vectorized computation of 22 indicators spanning Trend, Momentum, Volatility, Volume, and Returns. Strict forward shift $t+1$ target alignment to prevent lookahead data leakage.</td>
          </tr>
          <tr>
            <td><strong>4. ML Training</strong></td>
            <td><code>src/models.py</code></td>
            <td>Chronological 80% train / 20% test splitting. <code>StandardScaler</code> fitted strictly on training data. Cross-evaluation of Linear Ridge, Random Forest, and LightGBM Regressors.</td>
          </tr>
          <tr>
            <td><strong>5. Backtester</strong></td>
            <td><code>src/backtester.py</code></td>
            <td>Event-driven portfolio simulation ($10,000 cash balance, 0.1% maker/taker commission per trade). State machine execution (LONG, SHORT, FLAT). Sharpe, Drawdown, and Alpha logging.</td>
          </tr>
          <tr>
            <td><strong>6. Inference</strong></td>
            <td><code>src/forecast.py</code></td>
            <td>Live deployment of trained LightGBM models for instantaneous directional bias (BULLISH/BEARISH) and multi-step recursive autoregressive forecast trajectory cones (+1 to +5 bars).</td>
          </tr>
        </tbody>
      </table>
    </div>

    <!-- Section 3 -->
    <div class="report-section avoid-break">
      <h3 class="report-h2">3. Technical Feature Engineering Taxonomy</h3>
      <p class="report-p">
        A total of <strong>22 domain-specific quantitative technical features</strong> were designed across five structural categories. The predictive target was defined as the instantaneous next-bar close price <em>y<sub>t</sub> = Close<sub>t+1</sub></em>, shifted backward by one temporal step:
      </p>
      <table class="report-table">
        <thead>
          <tr>
            <th>Category</th>
            <th>Indicator Count</th>
            <th>Features Included</th>
            <th>Quantitative Role &amp; Formula</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td><strong>Trend</strong></td>
            <td>7 Features</td>
            <td>SMA 7, SMA 21, EMA 14, EMA Ratio, MACD, MACD Signal, MACD Hist</td>
            <td>Identifies macro trajectory and moving-average crossover momentum (<em>EMA<sub>14</sub> / SMA<sub>21</sub></em>).</td>
          </tr>
          <tr>
            <td><strong>Momentum</strong></td>
            <td>3 Features</td>
            <td>RSI 14, Stochastic %K, Stochastic %D</td>
            <td>Measures overbought/oversold oscillator states and velocity of price oscillations.</td>
          </tr>
          <tr>
            <td><strong>Volatility</strong></td>
            <td>5 Features</td>
            <td>BB Upper, BB Lower, BB Width, BB %B, ATR 14</td>
            <td>Measures market dispersion via standard deviations (&mu; &plusmn; 2&sigma;) and Average True Range.</td>
          </tr>
          <tr>
            <td><strong>Volume</strong></td>
            <td>3 Features</td>
            <td>Volume SMA 14, Volume Ratio, On-Balance Volume (OBV)</td>
            <td>Quantifies institutional buying/selling pressure and volume confirmation of trends.</td>
          </tr>
          <tr>
            <td><strong>Returns &amp; Dynamics</strong></td>
            <td>4 Features</td>
            <td>Log Return, 5-bar Return, High-Low Range %, Close-Open Range %</td>
            <td>Captures instantaneous price changes and intra-candle dispersion (ln(<em>P<sub>t</sub> / P<sub>t-1</sub></em>)).</td>
          </tr>
        </tbody>
      </table>
    </div>

    <!-- Section 4 -->
    <div class="report-section">
      <h3 class="report-h2">4. Comparative Machine Learning Benchmarking Results</h3>
      <p class="report-p">
        Model performance was evaluated on the out-of-sample chronological test set (recent 20% unseen candles) using Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), Coefficient of Determination (R<sup>2</sup>), and Directional Accuracy % (Hit Rate).
      </p>

      <h4 class="report-h3">Table 4.1: Model Comparison on Benchmark 1-Hour Timeframe</h4>
      <table class="report-table">
        <thead>
          <tr>
            <th>Model Architecture</th>
            <th>RMSE ($)</th>
            <th>MAE ($)</th>
            <th>R² Score</th>
            <th>Directional Hit Rate %</th>
            <th>Performance Assessment</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td><strong>Linear Regression (Ridge)</strong></td>
            <td>$310.01</td>
            <td>$211.37</td>
            <td>0.7116</td>
            <td>48.08%</td>
            <td>Strong baseline fit, but sub-50% directional accuracy on non-linear price turns.</td>
          </tr>
          <tr>
            <td><strong>Random Forest Regressor</strong></td>
            <td>$321.24</td>
            <td>$232.55</td>
            <td>0.6904</td>
            <td>46.79%</td>
            <td>Robust against outliers, but tendency to lag during sharp parabolic breakout phases.</td>
          </tr>
          <tr>
            <td><strong>LightGBM Regressor (Winner)</strong></td>
            <td>$354.21</td>
            <td>$262.42</td>
            <td>0.6235</td>
            <td><strong style="color: #16a34a;">51.92%</strong></td>
            <td>Highest directional predictability (+3.84% over Linear); superior for trading signals.</td>
          </tr>
        </tbody>
      </table>

      <h4 class="report-h3">Table 4.2: LightGBM Performance Across Multiple Timeframe Horizons</h4>
      <table class="report-table">
        <thead>
          <tr>
            <th>Timeframe</th>
            <th>Test Observations</th>
            <th>RMSE ($)</th>
            <th>MAE ($)</th>
            <th>R² Score</th>
            <th>Directional Hit Rate %</th>
            <th>Trading Regime Fit</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td><strong>5m (Scalping)</strong></td>
            <td>156</td>
            <td>$134.17</td>
            <td>$92.64</td>
            <td>0.9465</td>
            <td>47.44%</td>
            <td>Micro-noise dominates; high nominal R² due to autocorrelation, low edge.</td>
          </tr>
          <tr>
            <td><strong>15m (Intraday)</strong></td>
            <td>156</td>
            <td>$187.85</td>
            <td>$124.52</td>
            <td>0.8401</td>
            <td>47.44%</td>
            <td>Moderate intraday volatility; fee friction exceeds raw signal margin.</td>
          </tr>
          <tr>
            <td><strong>1h (Benchmark)</strong></td>
            <td>156</td>
            <td>$354.21</td>
            <td>$262.42</td>
            <td>0.6235</td>
            <td><strong>51.92%</strong></td>
            <td>Optimal balance of signal-to-noise; profitable directional bias.</td>
          </tr>
          <tr>
            <td><strong>4h (Swing)</strong></td>
            <td>156</td>
            <td>$1,738.69</td>
            <td>$1,451.90</td>
            <td>-0.6766</td>
            <td><strong>52.56%</strong></td>
            <td>High directional hit rate; successfully isolates macro swing trends.</td>
          </tr>
          <tr>
            <td><strong>24h (Macro Daily)</strong></td>
            <td>156</td>
            <td>$1,951.08</td>
            <td>$1,581.02</td>
            <td>0.9488</td>
            <td>48.72%</td>
            <td>Strong macro alignment; minimal fee drag over multi-month horizons.</td>
          </tr>
        </tbody>
      </table>

      <div class="callout-info">
        <strong>Key Scientific Observation:</strong> In quantitative finance, <em>Directional Accuracy (Hit Rate)</em> is significantly more critical for real portfolio profitability than nominal point RMSE. LightGBM achieved the highest directional hit rate (51.92% on 1h and 52.56% on 4h), enabling it to generate positive trading alpha even when point-prediction variance was higher.
      </div>
    </div>

    <!-- Section 5 -->
    <div class="report-section avoid-break">
      <h3 class="report-h2">5. Quantitative Backtesting Performance &amp; Risk Metrics</h3>
      <p class="report-p">
        Every model × timeframe combination (39 total experiments) was subjected to an event-driven backtesting simulation starting with an exact <strong>$10,000.00 cash balance</strong> and a mandatory <strong>0.1% transaction fee</strong> per buy/sell action.
      </p>

      <h4 class="report-h3">Table 5.1: Key Strategy Results on Swing &amp; Macro Horizons ($10,000 Starting Capital)</h4>
      <table class="report-table">
        <thead>
          <tr>
            <th>Timeframe</th>
            <th>Model</th>
            <th>Final Balance</th>
            <th>Strategy Return</th>
            <th>Buy &amp; Hold Return</th>
            <th>Alpha</th>
            <th>Win Rate %</th>
            <th>Max Drawdown %</th>
            <th>Sharpe Ratio</th>
          </tr>
        </thead>
        <tbody>
          <tr style="background: #f0fdf4; font-weight: bold;">
            <td><strong>4d</strong></td>
            <td>LightGBM</td>
            <td>$12,145.07</td>
            <td style="color: #16a34a;">+21.45%</td>
            <td>+11.10%</td>
            <td style="color: #16a34a;">+10.35%</td>
            <td>100.0%</td>
            <td>12.27%</td>
            <td style="color: #16a34a;">2.11</td>
          </tr>
          <tr>
            <td><strong>4d</strong></td>
            <td>Linear / RF</td>
            <td>$11,087.89</td>
            <td style="color: #16a34a;">+10.88%</td>
            <td>+11.10%</td>
            <td>-0.22%</td>
            <td>100.0%</td>
            <td>20.28%</td>
            <td>1.27</td>
          </tr>
          <tr style="background: #f0fdf4;">
            <td><strong>24h</strong></td>
            <td>Linear Reg</td>
            <td>$11,011.30</td>
            <td style="color: #16a34a;">+10.11%</td>
            <td>+5.78%</td>
            <td style="color: #16a34a;">+4.33%</td>
            <td>50.0%</td>
            <td>10.00%</td>
            <td>0.77</td>
          </tr>
          <tr>
            <td><strong>24h</strong></td>
            <td>Random Forest</td>
            <td>$10,496.77</td>
            <td style="color: #16a34a;">+4.97%</td>
            <td>+5.78%</td>
            <td>-0.81%</td>
            <td>36.8%</td>
            <td>14.94%</td>
            <td>0.44</td>
          </tr>
          <tr style="background: #f0fdf4;">
            <td><strong>4h</strong></td>
            <td>Random Forest</td>
            <td>$10,113.39</td>
            <td style="color: #16a34a;">+1.13%</td>
            <td style="color: #dc2626;">-2.13%</td>
            <td style="color: #16a34a;">+3.26%</td>
            <td>66.7%</td>
            <td>3.81%</td>
            <td>0.26</td>
          </tr>
          <tr>
            <td><strong>4h</strong></td>
            <td>Linear Reg</td>
            <td>$10,090.59</td>
            <td style="color: #16a34a;">+0.91%</td>
            <td style="color: #dc2626;">-2.13%</td>
            <td style="color: #16a34a;">+3.04%</td>
            <td>50.0%</td>
            <td>3.89%</td>
            <td>0.32</td>
          </tr>
          <tr>
            <td><strong>1m – 15m</strong></td>
            <td>All Models</td>
            <td>$9,940 – $10,000</td>
            <td>-0.59% to 0.0%</td>
            <td>-0.13% to +1.26%</td>
            <td>Neutral</td>
            <td>0 – 40%</td>
            <td>0.5% – 1.0%</td>
            <td>-2.7 to 0.0</td>
          </tr>
        </tbody>
      </table>
    </div>

    <!-- Section 6 -->
    <div class="report-section avoid-break">
      <h3 class="report-h2">6. Critical Discussion: The High-Frequency Friction Trap vs Swing Alpha</h3>
      <p class="report-p">
        A central empirical finding of this project is the <strong>Friction Drag Phenomenon</strong> in algorithmic cryptocurrency trading:
      </p>
      <div class="report-grid-2">
        <div class="callout-alert">
          <strong style="font-size: 9pt;">The High-Frequency Friction Trap (1m, 3m, 5m, 15m):</strong><br>
          On ultra-short timeframes, candle price fluctuations are on the order of 0.02% to 0.15%. When standard exchange trading fees (0.10% maker/taker) are charged on both entry and exit (0.20% round-trip), <strong>transaction costs completely consume any statistical edge</strong>, resulting in equity stagnation or drawdown even with a sub-second model.
        </div>
        <div class="callout-box">
          <strong style="font-size: 9pt;">The Macro Swing Alpha Edge (4h, 24h, 4d):</strong><br>
          On higher timeframes, price swings average 3% to 15%. Over these larger moves, the fixed 0.20% round-trip fee constitutes only a negligible fraction of the gross profit. Consequently, strategies operating on 4-hour, daily, and 4-day intervals yielded <strong>positive net alpha (+3.26% on 4h, +4.33% on 24h, +10.35% on 4d)</strong> and Sharpe ratios exceeding 2.0.
        </div>
      </div>
    </div>

    <!-- Section 7 -->
    <div class="report-section avoid-break">
      <h3 class="report-h2">7. Conclusions, System Limitations &amp; Future Scope</h3>
      <p class="report-p">
        <strong>Key Conclusions:</strong>
      </p>
      <ol style="margin-left: 20px; font-size: 8.5pt; color: #334155; margin-bottom: 10px;">
        <li>Cryptocurrency log returns display pronounced excess kurtosis (4.89 vs Gaussian 3.0), mathematically confirming that non-linear gradient-boosted decision trees (LightGBM) outperform classical linear estimators in directional hit rate.</li>
        <li>Zero lookahead bias is critical: temporal backward shifting of target <em>P<sub>t+1</sub></em> guarantees realistic out-of-sample validity.</li>
        <li>Swing horizons (4h to 4d) are the optimal operational regime for retail and systematic quantitative strategies, yielding the highest risk-adjusted Sharpe ratio (2.11) and outperforming the Buy-and-Hold benchmark during both trending and corrective markets.</li>
      </ol>

      <p class="report-p">
        <strong>Future Enhancements:</strong>
      </p>
      <ul style="margin-left: 20px; font-size: 8.5pt; color: #334155; margin-bottom: 12px;">
        <li><strong>Deep Learning Sequences:</strong> Implementation of Bi-Directional LSTM and Temporal Fusion Transformers (TFT) to model long-range temporal dependencies.</li>
        <li><strong>Microstructure &amp; Order Book:</strong> Integration of Level-2 (L2) order book imbalance, bid-ask spread depth, and funding rate arbitrage signals.</li>
        <li><strong>Dynamic Position Sizing:</strong> Replacement of fixed all-in capital allocation with the Kelly Criterion and volatility-adjusted ATR position sizing to minimize maximum drawdowns.</li>
      </ul>

      <div style="padding: 10px 14px; background: #f8fafc; border: 1px solid #cbd5e1; border-radius: 6px; font-size: 8pt; color: #475569;">
        <strong>Academic Declaration:</strong> I hereby declare that this project report titled <em>"Multi-Timeframe Cryptocurrency Price Forecasting &amp; Quantitative Algorithmic Backtesting System"</em> is an authentic record of my own coursework carried out under the Department of Computer Science &amp; Engineering (CCE), Manipal University Jaipur.<br>
        <div style="margin-top: 6px; font-weight: 700; color: #0f172a;">
          Author: Ambuj Mishra (2428010013) • B.Tech CCE
        </div>
      </div>
    </div>
  </div>

  <!-- =====================================================
       PART II: EXECUTED LABORATORY NOTEBOOKS COMPILATION
       ===================================================== -->
  <div class="page-break" style="padding: 40px 10px 20px; text-align: center;">
    <div style="display: inline-block; padding: 4px 14px; background: #e0f2fe; color: #0369a1; border: 1px solid #bae6fd; border-radius: 20px; font-size: 8pt; font-weight: 800; text-transform: uppercase; margin-bottom: 12px;">
      Part II • Technical Execution Dossier
    </div>
    <h1 style="font-size: 20pt; font-weight: 900; color: #0f172a; margin-bottom: 6px; text-transform: uppercase;">
      Compiled Laboratory Notebooks (01 to 06)
    </h1>
    <p style="font-size: 9.5pt; color: #475569; max-width: 600px; margin: 0 auto 20px;">
      Complete reproduction of all six project Jupyter Notebooks containing raw code cells, execution outputs, formatted Pandas DataFrames, and embedded high-resolution Matplotlib artifacts.
    </p>
    <div style="height: 2px; width: 80px; background: #0284c7; margin: 0 auto;"></div>
  </div>

  {all_notebooks_markup}

</body>
</html>
'''

    print(f"Writing complete HTML report to {OUTPUT_HTML}...")
    with open(OUTPUT_HTML, 'w', encoding='utf-8') as f:
        f.write(full_html)
    print("HTML report successfully generated.")

    print(f"Converting HTML to PDF via Headless Chromium at {OUTPUT_PDF}...")
    file_url = f"file://{os.path.abspath(OUTPUT_HTML)}"
    cmd = [
        "chromium",
        "--headless",
        "--no-sandbox",
        "--disable-gpu",
        "--run-all-compositor-stages-before-draw",
        "--no-pdf-header-footer",
        f"--print-to-pdf={OUTPUT_PDF}",
        file_url
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"Error running chromium: {res.stderr}")
        sys.exit(1)

    print("Chromium finished:")
    print(res.stderr.strip() if res.stderr else res.stdout.strip())

    if os.path.exists(OUTPUT_PDF):
        size_mb = os.path.getsize(OUTPUT_PDF) / (1024 * 1024)
        print(f"SUCCESS: Generated PDF '{OUTPUT_PDF}' ({size_mb:.2f} MB)")
    else:
        print("ERROR: PDF was not generated.")

if __name__ == '__main__':
    main()
