"""
HTML 报告生成器 v2

改进:
- 可排序表格 (点击表头升降序)
- 搜索/筛选 (每张表独立搜索框)
- 全量展示所有行 (滚动容器, 表头固定)
- Tab 导航 (模型报告多报表切换)
- 完整集成 ModelDeliveryReport.generate_full_report() 全部输出
- 自包含: 内联 CSS + JS + SVG, 无外部依赖
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
from datetime import datetime


PALETTE = {
    'bg': '#f6f9fc', 'surface': '#ffffff', 'border': '#e3e8ee',
    'text': '#1a1f36', 'text_secondary': '#697386',
    'primary': '#635bff', 'primary_light': '#a3a0ff',
    'success': '#0c9b6a', 'warning': '#e4a10e', 'danger': '#df1b41',
    'info': '#1a73e8',
}

FONT_STACK = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif"


def _css() -> str:
    return f"""
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
    body {{ font-family: {FONT_STACK}; background: {PALETTE['bg']}; color: {PALETTE['text']}; line-height: 1.6; -webkit-font-smoothing: antialiased; }}
    .container {{ max-width: 1280px; margin: 0 auto; padding: 32px 24px; }}
    .header {{ background: {PALETTE['surface']}; border: 1px solid {PALETTE['border']}; border-radius: 12px; padding: 32px; margin-bottom: 24px; box-shadow: 0 1px 3px rgba(0,0,0,0.04), 0 4px 12px rgba(0,0,0,0.03); }}
    .header h1 {{ font-size: 28px; font-weight: 700; margin-bottom: 8px; }}
    .header .meta {{ color: {PALETTE['text_secondary']}; font-size: 14px; }}
    .header .meta span {{ margin-right: 16px; }}
    .section {{ background: {PALETTE['surface']}; border: 1px solid {PALETTE['border']}; border-radius: 12px; padding: 24px; margin-bottom: 24px; box-shadow: 0 1px 3px rgba(0,0,0,0.04), 0 4px 12px rgba(0,0,0,0.03); }}
    .section-title {{ font-size: 20px; font-weight: 600; margin-bottom: 16px; padding-bottom: 12px; border-bottom: 2px solid {PALETTE['primary']}; display: flex; align-items: center; gap: 8px; }}
    .section-title .badge {{ font-size: 12px; font-weight: 500; padding: 2px 10px; border-radius: 12px; background: {PALETTE['primary']}; color: white; }}
    .subsection-title {{ font-size: 16px; font-weight: 600; margin: 20px 0 12px; color: {PALETTE['text']}; }}

    /* 表格容器: 可滚动 + 固定表头 */
    .table-wrapper {{ position: relative; margin-bottom: 16px; }}
    .table-toolbar {{ display: flex; align-items: center; gap: 12px; margin-bottom: 8px; }}
    .table-search {{ padding: 6px 12px; border: 1px solid {PALETTE['border']}; border-radius: 6px; font-size: 13px; width: 240px; outline: none; transition: border-color 0.2s; }}
    .table-search:focus {{ border-color: {PALETTE['primary']}; box-shadow: 0 0 0 3px rgba(99,91,255,0.1); }}
    .table-info {{ font-size: 12px; color: {PALETTE['text_secondary']}; }}
    .table-scroll {{ max-height: 600px; overflow-y: auto; border: 1px solid {PALETTE['border']}; border-radius: 8px; }}
    .table-scroll::-webkit-scrollbar {{ width: 8px; height: 8px; }}
    .table-scroll::-webkit-scrollbar-thumb {{ background: #cdd5e0; border-radius: 4px; }}
    .table-scroll::-webkit-scrollbar-track {{ background: {PALETTE['bg']}; }}

    table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    thead {{ position: sticky; top: 0; z-index: 10; }}
    th {{ background: {PALETTE['bg']}; font-weight: 600; text-align: left; padding: 10px 12px; border-bottom: 2px solid {PALETTE['border']}; white-space: nowrap; color: {PALETTE['text_secondary']}; font-size: 12px; text-transform: uppercase; letter-spacing: 0.5px; cursor: pointer; user-select: none; position: relative; }}
    th:hover {{ background: #e8ecf3; color: {PALETTE['primary']}; }}
    th.sort-asc::after {{ content: ' ▲'; color: {PALETTE['primary']}; font-size: 10px; }}
    th.sort-desc::after {{ content: ' ▼'; color: {PALETTE['primary']}; font-size: 10px; }}
    td {{ padding: 8px 12px; border-bottom: 1px solid {PALETTE['border']}; white-space: nowrap; max-width: 300px; overflow: hidden; text-overflow: ellipsis; }}
    tr:nth-child(even) td {{ background: #fafbfc; }}
    tr:hover td {{ background: #f0f4ff; }}

    .metric-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; margin-bottom: 20px; }}
    .metric-card {{ background: {PALETTE['surface']}; border: 1px solid {PALETTE['border']}; border-radius: 10px; padding: 20px; text-align: center; }}
    .metric-card .label {{ font-size: 12px; color: {PALETTE['text_secondary']}; text-transform: uppercase; letter-spacing: 0.5px; }}
    .metric-card .value {{ font-size: 32px; font-weight: 700; margin: 8px 0; }}
    .metric-card .sub {{ font-size: 12px; color: {PALETTE['text_secondary']}; }}
    .metric-card.good .value {{ color: {PALETTE['success']}; }}
    .metric-card.bad .value {{ color: {PALETTE['danger']}; }}
    .metric-card.neutral .value {{ color: {PALETTE['primary']}; }}
    .metric-card.warning .value {{ color: {PALETTE['warning']}; }}

    .bar-chart {{ margin: 12px 0; }}
    .bar-row {{ display: flex; align-items: center; gap: 8px; margin-bottom: 6px; font-size: 13px; }}
    .bar-label {{ width: 140px; text-align: right; color: {PALETTE['text_secondary']}; flex-shrink: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
    .bar-track {{ flex: 1; height: 22px; background: {PALETTE['bg']}; border-radius: 4px; overflow: hidden; }}
    .bar-fill {{ height: 100%; border-radius: 4px; display: flex; align-items: center; justify-content: flex-end; padding-right: 8px; font-size: 11px; color: white; font-weight: 600; }}
    .bar-fill.good {{ background: linear-gradient(90deg, {PALETTE['success']}, #15b37d); }}
    .bar-fill.warning {{ background: linear-gradient(90deg, {PALETTE['warning']}, #f0b429); }}
    .bar-fill.bad {{ background: linear-gradient(90deg, {PALETTE['danger']}, #e8456a); }}
    .bar-fill.neutral {{ background: linear-gradient(90deg, {PALETTE['primary']}, {PALETTE['primary_light']}); }}

    .tag {{ display: inline-block; padding: 2px 10px; border-radius: 12px; font-size: 12px; font-weight: 500; }}
    .tag.success {{ background: #e6f7ef; color: {PALETTE['success']}; }}
    .tag.danger {{ background: #fce8ec; color: {PALETTE['danger']}; }}
    .tag.warning {{ background: #fef5e3; color: {PALETTE['warning']}; }}
    .tag.info {{ background: #e8f0fe; color: {PALETTE['info']}; }}

    .chart-container {{ margin: 16px 0; text-align: center; }}
    .chart-container svg {{ max-width: 100%; height: auto; }}
    .two-col {{ display: grid; grid-template-columns: 1fr 1fr; gap: 24px; }}
    .three-col {{ display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 16px; }}
    .footer {{ text-align: center; color: {PALETTE['text_secondary']}; font-size: 12px; padding: 24px 0; }}
    .summary-box {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 12px; margin-bottom: 16px; }}
    .summary-item {{ background: {PALETTE['bg']}; border-radius: 8px; padding: 16px; text-align: center; border: 1px solid {PALETTE['border']}; }}
    .summary-item .num {{ font-size: 24px; font-weight: 700; color: {PALETTE['primary']}; }}
    .summary-item .desc {{ font-size: 12px; color: {PALETTE['text_secondary']}; margin-top: 4px; }}

    /* Tab 导航 */
    .tab-nav {{ display: flex; gap: 0; border-bottom: 2px solid {PALETTE['border']}; margin-bottom: 20px; overflow-x: auto; }}
    .tab-btn {{ padding: 10px 20px; border: none; background: none; cursor: pointer; font-size: 14px; font-weight: 500; color: {PALETTE['text_secondary']}; border-bottom: 2px solid transparent; margin-bottom: -2px; white-space: nowrap; transition: all 0.2s; }}
    .tab-btn:hover {{ color: {PALETTE['primary']}; }}
    .tab-btn.active {{ color: {PALETTE['primary']}; border-bottom-color: {PALETTE['primary']}; }}
    .tab-content {{ display: none; }}
    .tab-content.active {{ display: block; }}

    /* 参数表 */
    .params-table td:first-child {{ font-weight: 600; width: 240px; color: {PALETTE['text_secondary']}; }}
    .params-table td {{ font-family: 'SF Mono', 'Fira Code', monospace; font-size: 12px; }}

    .no-data {{ color: {PALETTE['text_secondary']}; font-size: 14px; padding: 24px; text-align: center; }}
"""


def _js() -> str:
    """返回 JavaScript: 表格排序 + 搜索 + Tab 切换"""
    return """
    // ===== 表格排序 =====
    function sortTable(tableId, colIdx, dataType) {
        const table = document.getElementById(tableId);
        if (!table) return;
        const tbody = table.querySelector('tbody');
        const rows = Array.from(tbody.querySelectorAll('tr'));
        const th = table.querySelectorAll('th')[colIdx];

        // 切换排序方向
        const isAsc = th.classList.contains('sort-asc');
        // 清除所有排序标记
        table.querySelectorAll('th').forEach(h => h.classList.remove('sort-asc', 'sort-desc'));
        th.classList.add(isAsc ? 'sort-desc' : 'sort-asc');

        const dir = isAsc ? -1 : 1;
        rows.sort((a, b) => {
            let va = a.cells[colIdx] ? a.cells[colIdx].textContent.trim() : '';
            let vb = b.cells[colIdx] ? b.cells[colIdx].textContent.trim() : '';
            // 尝试数值比较
            const na = parseFloat(va.replace(/[%,,]/g, ''));
            const nb = parseFloat(vb.replace(/[%,,]/g, ''));
            if (!isNaN(na) && !isNaN(nb) && va !== '' && vb !== '') {
                return (na - nb) * dir;
            }
            return va.localeCompare(vb, 'zh') * dir;
        });
        rows.forEach(r => tbody.appendChild(r));
    }

    // ===== 表格搜索 =====
    function filterTable(inputId, tableId) {
        const input = document.getElementById(inputId);
        const table = document.getElementById(tableId);
        if (!input || !table) return;
        const query = input.value.toLowerCase().trim();
        const rows = table.querySelectorAll('tbody tr');
        let visible = 0;
        rows.forEach(row => {
            const text = row.textContent.toLowerCase();
            if (text.includes(query)) {
                row.style.display = '';
                visible++;
            } else {
                row.style.display = 'none';
            }
        });
        // 更新计数
        const info = document.getElementById(tableId + '_info');
        if (info) info.textContent = `显示 ${visible} / ${rows.length} 行`;
    }

    // ===== Tab 切换 =====
    function switchTab(tabGroup, tabId) {
        // 隐藏所有 tab content
        document.querySelectorAll(`[data-tab-group="${tabGroup}"]`).forEach(el => {
            el.classList.remove('active');
        });
        // 显示选中的 tab
        document.getElementById(tabId).classList.add('active');
        // 更新按钮状态
        document.querySelectorAll(`[data-tab-btn-group="${tabGroup}"]`).forEach(btn => {
            btn.classList.remove('active');
        });
        document.querySelector(`[data-tab-btn-group="${tabGroup}"][data-tab-target="${tabId}"]`).classList.add('active');
    }
"""


def _table_html(df: pd.DataFrame, table_id: str = None, show_all: bool = True,
                max_rows: int = None, searchable: bool = True) -> str:
    """
    渲染可排序、可搜索的 HTML 表格

    Args:
        df: 数据框
        table_id: 表格 DOM id
        show_all: 是否展示所有行 (True=全量, False=截断到 max_rows)
        max_rows: 当 show_all=False 时的最大行数
        searchable: 是否显示搜索框
    """
    if df is None or df.empty:
        return '<div class="no-data">暂无数据</div>'

    if table_id is None:
        table_id = f"tbl_{id(df)}"

    display_df = df.copy() if show_all else df.head(max_rows or 50).copy()

    # 数值格式化
    for col in display_df.columns:
        if display_df[col].dtype in ['float64', 'float32']:
            display_df[col] = display_df[col].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "-")

    # 表头
    headers = ""
    for i, col in enumerate(display_df.columns):
        headers += f'<th onclick="sortTable(\'{table_id}\', {i})">{col}</th>'

    # 表体
    rows_html = ""
    for _, row in display_df.iterrows():
        cells = ""
        for col in display_df.columns:
            val = row[col]
            if pd.isna(val):
                val = "-"
            cells += f"<td title='{val}'>{val}</td>"
        rows_html += f"<tr>{cells}</tr>"

    # 搜索框
    search_html = ""
    if searchable:
        search_html = f"""
        <div class="table-toolbar">
            <input type="text" class="table-search" placeholder="搜索..." id="{table_id}_search"
                   oninput="filterTable('{table_id}_search', '{table_id}')">
            <span class="table-info" id="{table_id}_info">共 {len(display_df)} 行</span>
        </div>"""

    truncated = ""
    if not show_all and len(df) > len(display_df):
        truncated = f'<span class="table-info">显示前 {len(display_df)} 行（共 {len(df)} 行）</span>'

    return f"""
    <div class="table-wrapper">
        {search_html}
        {truncated}
        <div class="table-scroll">
            <table id="{table_id}">
                <thead><tr>{headers}</tr></thead>
                <tbody>{rows_html}</tbody>
            </table>
        </div>
    </div>"""


def _bar_chart_html(items, color='neutral', max_val=None):
    if not items:
        return '<div class="no-data">暂无数据</div>'
    if max_val is None:
        max_val = max((v for _, v in items if v is not None and not pd.isna(v)), default=1) or 1
    bars = ""
    for label, val in items:
        if val is None or pd.isna(val):
            val = 0
        width = (val / max_val * 100) if max_val > 0 else 0
        bars += f'<div class="bar-row"><div class="bar-label" title="{label}">{label}</div><div class="bar-track"><div class="bar-fill {color}" style="width:{width:.1f}%">{val:.4f}</div></div></div>'
    return f'<div class="bar-chart">{bars}</div>'


def _ks_curve_svg(y_true, y_score, width=480, height=300):
    try:
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(y_true, y_score)
        ks = max(tpr - fpr)
        n = len(fpr)
        xs, ys = width - 60, height - 60
        good_path = " ".join(f"{30 + fpr[i]*xs:.1f},{30 + (1-fpr[i])*ys:.1f}" for i in range(n))
        bad_path = " ".join(f"{30 + fpr[i]*xs:.1f},{30 + (1-tpr[i])*ys:.1f}" for i in range(n))
        ki = np.argmax(tpr - fpr)
        kx = 30 + fpr[ki] * xs
        return f"""<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg">
            <rect x="30" y="30" width="{xs}" height="{ys}" fill="#f6f9fc" stroke="#e3e8ee"/>
            <text x="{width//2}" y="20" text-anchor="middle" font-size="13" font-weight="600" fill="#1a1f36">KS Curve (KS={ks:.4f})</text>
            <polyline points="{good_path}" fill="none" stroke="#0c9b6a" stroke-width="2"/>
            <polyline points="{bad_path}" fill="none" stroke="#df1b41" stroke-width="2"/>
            <line x1="{kx}" y1="{30+(1-fpr[ki])*ys:.1f}" x2="{kx}" y2="{30+(1-tpr[ki])*ys:.1f}" stroke="#e4a10e" stroke-width="2" stroke-dasharray="4"/>
        </svg>"""
    except:
        return ""


def _roc_curve_svg(y_true, y_score, width=480, height=300):
    try:
        from sklearn.metrics import roc_curve, auc
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        n = len(fpr)
        xs, ys = width - 60, height - 60
        path = " ".join(f"{30 + fpr[i]*xs:.1f},{30 + (1-tpr[i])*ys:.1f}" for i in range(n))
        fill = f"30,{30+ys} {path} {30+fpr[-1]*xs:.1f},{30+ys}"
        return f"""<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg">
            <rect x="30" y="30" width="{xs}" height="{ys}" fill="#f6f9fc" stroke="#e3e8ee"/>
            <text x="{width//2}" y="20" text-anchor="middle" font-size="13" font-weight="600" fill="#1a1f36">ROC Curve (AUC={roc_auc:.4f})</text>
            <polygon points="{fill}" fill="rgba(99,91,255,0.1)"/>
            <polyline points="30,{30+ys} {30+xs},30" fill="none" stroke="#697386" stroke-width="1" stroke-dasharray="4"/>
            <polyline points="{path}" fill="none" stroke="#635bff" stroke-width="2"/>
        </svg>"""
    except:
        return ""


def _lift_chart_svg(y_true, y_score, n_bins=10, width=480, height=300):
    try:
        df = pd.DataFrame({'y': y_true, 's': y_score}).dropna()
        if len(df) == 0:
            return ""
        baseline = df['y'].mean()
        df['bin'] = pd.qcut(df['s'], q=n_bins, labels=False, duplicates='drop') + 1
        lift_vals = df.groupby('bin')['y'].mean() / baseline if baseline > 0 else df.groupby('bin')['y'].mean()
        bins = sorted(lift_vals.index, reverse=True)
        vals = [lift_vals[b] for b in bins]
        bw = (width - 60) / len(vals) * 0.7
        gap = (width - 60) / len(vals) * 0.3
        max_val = max(vals) if vals else 1
        ys = height - 60
        bars = ""
        for i, v in enumerate(vals):
            x = 30 + i * (bw + gap) + gap / 2
            h = (v / max_val) * ys * 0.85
            y = 30 + ys - h
            color = '#df1b41' if v > 1 else '#0c9b6a'
            bars += f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw:.1f}" height="{h:.1f}" fill="{color}" opacity="0.8" rx="2"/>'
            bars += f'<text x="{x+bw/2:.1f}" y="{y-4:.1f}" text-anchor="middle" font-size="9" fill="#697386">{v:.2f}</text>'
        return f"""<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg">
            <rect x="30" y="30" width="{width-60}" height="{ys}" fill="#f6f9fc" stroke="#e3e8ee"/>
            <text x="{width//2}" y="20" text-anchor="middle" font-size="13" font-weight="600" fill="#1a1f36">Lift by Decile</text>
            {bars}
        </svg>"""
    except:
        return ""


def _html_template(title: str, sections_html: str, now: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title><style>{_css()}</style></head>
<body><div class="container">
<div class="header"><h1>{title}</h1><div class="meta"><span>生成时间: {now}</span></div></div>
{sections_html}
<div class="footer">Powered by model_tools HTML Report Generator v2</div>
</div>
<script>{_js()}</script>
</body></html>"""


# ============================================================
# 数据分析 HTML 报告
# ============================================================

def generate_data_analysis_html(analyzer, features=None, save_path=None, title="数据分析报告"):
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    sections = []
    total_rows = len(analyzer.data)
    total_cols = len(analyzer.data.columns)
    numeric_count = len(analyzer._numeric_cols)
    categorical_count = len(analyzer._categorical_cols)

    # 概览
    sections.append(("数据概览", f"""
    <div class="metric-grid">
        <div class="metric-card neutral"><div class="label">总样本数</div><div class="value">{total_rows:,}</div></div>
        <div class="metric-card neutral"><div class="label">总特征数</div><div class="value">{total_cols}</div></div>
        <div class="metric-card good"><div class="label">数值特征</div><div class="value">{numeric_count}</div></div>
        <div class="metric-card warning"><div class="label">分类特征</div><div class="value">{categorical_count}</div></div>
    </div>"""))

    # 缺失值分析 - 全量展示 + 可排序
    try:
        missing_df = analyzer.missing_summary(features)
        if not missing_df.empty:
            missing_rates = missing_df['缺失率'].astype(float)
            high_missing = int((missing_rates > 0.5).sum())
            med_missing = int(((missing_rates > 0.1) & (missing_rates <= 0.5)).sum())
            low_missing = int((missing_rates <= 0.1).sum())

            summary_html = f"""
            <div class="summary-box">
                <div class="summary-item"><div class="num">{high_missing}</div><div class="desc">缺失率 > 50%</div></div>
                <div class="summary-item"><div class="num">{med_missing}</div><div class="desc">缺失率 10%-50%</div></div>
                <div class="summary-item"><div class="num">{low_missing}</div><div class="desc">缺失率 <= 10%</div></div>
            </div>"""
            top_missing = missing_df.head(20)
            items = [(row['特征名'], float(row['缺失率'])) for _, row in top_missing.iterrows()]
            bars = f"<div class='subsection-title'>Top 20 缺失率特征</div>{_bar_chart_html(items, 'bad')}"
            table = f"<div class='subsection-title'>缺失值明细 (全量, 可排序/搜索)</div>{_table_html(missing_df, 'tbl_missing', show_all=True)}"
            sections.append(("缺失值分析", summary_html + bars + table))
    except Exception as e:
        sections.append(("缺失值分析", f'<div class="no-data">分析失败: {e}</div>'))

    # 数值分布 - 全量
    try:
        dist_df = analyzer.distribution_summary(features)
        if not dist_df.empty:
            table = f"<div class='subsection-title'>数值分布统计 (全量, 可排序/搜索)</div>{_table_html(dist_df, 'tbl_dist', show_all=True)}"
            sections.append(("数值分布统计", table))
    except Exception as e:
        sections.append(("数值分布统计", f'<div class="no-data">分析失败: {e}</div>'))

    # 异常值 - 全量
    try:
        outlier_df = analyzer.outlier_summary(features)
        if not outlier_df.empty:
            outlier_df['_rate'] = outlier_df['异常值率'].astype(float)
            top_outliers = outlier_df.nlargest(20, '_rate')
            items = [(row['特征名'], float(row['异常值率'])) for _, row in top_outliers.iterrows()]
            bars = f"<div class='subsection-title'>Top 20 异常值率特征</div>{_bar_chart_html(items, 'warning')}"
            table = f"<div class='subsection-title'>异常值明细 (全量, 可排序/搜索)</div>{_table_html(outlier_df.drop('_rate', axis=1), 'tbl_outlier', show_all=True)}"
            sections.append(("异常值分析", bars + table))
    except Exception as e:
        sections.append(("异常值分析", f'<div class="no-data">分析失败: {e}</div>'))

    # 分类特征
    try:
        cat_df = analyzer.categorical_summary()
        if not cat_df.empty:
            sections.append(("分类特征统计", _table_html(cat_df, 'tbl_cat', show_all=True)))
    except Exception as e:
        sections.append(("分类特征统计", f'<div class="no-data">分析失败: {e}</div>'))

    sections_html = ""
    for i, (s_title, content) in enumerate(sections):
        sections_html += f'<div class="section"><div class="section-title"><span class="badge">{i+1}</span> {s_title}</div>{content}</div>'

    html = _html_template(title, sections_html, now)
    if save_path:
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(html)
        print(f"✓ 数据分析 HTML 报告已保存: {save_path}")
    return html


# ============================================================
# 模型评估 HTML 报告 (完整集成 generate_full_report)
# ============================================================

def generate_model_report_html(
    report_data: Dict[str, Any] = None,
    y_true_train=None, y_score_train=None,
    y_true_oos=None, y_score_oos=None,
    best_params: Dict = None,
    tuning_history: pd.DataFrame = None,
    feature_importance: pd.DataFrame = None,
    save_path: str = None,
    title: str = "模型交付报告"
):
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    report_data = report_data or {}
    sections = []
    tab_counter = [0]

    def make_tab_nav(tabs: List[tuple], group: str) -> str:
        """生成 Tab 导航按钮"""
        btns = ""
        for i, (tab_id, tab_label) in enumerate(tabs):
            cls = "tab-btn active" if i == 0 else "tab-btn"
            btns += f'<button class="{cls}" data-tab-btn-group="{group}" data-tab-target="{tab_id}" onclick="switchTab(\'{group}\', \'{tab_id}\')">{tab_label}</button>'
        return f'<div class="tab-nav">{btns}</div>'

    # ===== 1. 调参概览 =====
    if tuning_history is not None and not tuning_history.empty:
        n_trials = len(tuning_history)
        best_val_auc = tuning_history.get('val_auc', pd.Series()).max()
        params_html = ""
        if best_params:
            params_html = '<table class="params-table"><tbody>'
            for k, v in sorted(best_params.items()):
                params_html += f"<tr><td>{k}</td><td>{v}</td></tr>"
            params_html += '</tbody></table>'
        tuning_html = f"""
        <div class="summary-box">
            <div class="summary-item"><div class="num">{n_trials}</div><div class="desc">调参试验数</div></div>
            <div class="summary-item"><div class="num">{best_val_auc:.4f}</div><div class="desc">最佳验证集 AUC</div></div>
            <div class="summary-item"><div class="num">{len(best_params) if best_params else 0}</div><div class="desc">参数数量</div></div>
        </div>
        <div class="subsection-title">最优参数</div>{params_html}
        <div class="subsection-title">调参历史 (可排序/搜索)</div>{_table_html(tuning_history.drop(columns=['params'], errors='ignore'), 'tbl_tuning', show_all=True)}
        """
        sections.append(("调参概览", tuning_html))

    # ===== 2. 模型效果核心指标 + 图表 =====
    charts_and_metrics = ""
    if y_true_train is not None and y_score_train is not None:
        from sklearn.metrics import roc_auc_score, roc_curve
        train_auc = roc_auc_score(y_true_train, y_score_train)
        train_ks = max(roc_curve(y_true_train, y_score_train)[1] - roc_curve(y_true_train, y_score_train)[0])
        oos_auc = roc_auc_score(y_true_oos, y_score_oos) if y_true_oos is not None else None
        oos_ks = max(roc_curve(y_true_oos, y_score_oos)[1] - roc_curve(y_true_oos, y_score_oos)[0]) if y_true_oos is not None else None
        auc_gap = abs(train_auc - oos_auc) if oos_auc is not None else None
        ks_gap = abs(train_ks - oos_ks) if oos_ks is not None else None
        gap_tag = "success" if (auc_gap is not None and auc_gap <= 0.05) else "danger"

        oos_auc_s = f"{oos_auc:.4f}" if oos_auc is not None else "N/A"
        oos_ks_s = f"{oos_ks:.4f}" if oos_ks is not None else "N/A"
        auc_gap_s = f"{auc_gap:.4f}" if auc_gap is not None else "N/A"
        ks_gap_s = f"{ks_gap:.4f}" if ks_gap is not None else "N/A"

        charts_and_metrics = f"""
        <div class="metric-grid">
            <div class="metric-card good"><div class="label">训练集 AUC</div><div class="value">{train_auc:.4f}</div><div class="sub">KS: {train_ks:.4f}</div></div>
            <div class="metric-card {'good' if oos_auc and oos_auc >= 0.7 else 'warning'}"><div class="label">OOS AUC</div><div class="value">{oos_auc_s}</div><div class="sub">KS: {oos_ks_s}</div></div>
            <div class="metric-card {'good' if auc_gap and auc_gap <= 0.05 else 'bad'}"><div class="label">AUC 差距</div><div class="value">{auc_gap_s}</div><div class="sub">阈值 0.05</div></div>
            <div class="metric-card {'good' if ks_gap and ks_gap <= 0.03 else 'warning'}"><div class="label">KS 差距</div><div class="value">{ks_gap_s}</div><div class="sub">阈值 0.03</div></div>
        </div>
        <div style="text-align:center;margin-bottom:20px;"><span class="tag {gap_tag}">{'过拟合检查: 通过' if gap_tag == 'success' else '过拟合检查: 需关注'}</span></div>
        <div class="subsection-title">Train 评估图表</div>
        <div class="three-col">
            <div class="chart-container">{_ks_curve_svg(y_true_train, y_score_train, width=400, height=280)}</div>
            <div class="chart-container">{_roc_curve_svg(y_true_train, y_score_train, width=400, height=280)}</div>
            <div class="chart-container">{_lift_chart_svg(y_true_train, y_score_train, width=400, height=280)}</div>
        </div>"""

        if y_true_oos is not None and y_score_oos is not None:
            charts_and_metrics += f"""
        <div class="subsection-title">OOS 评估图表</div>
        <div class="three-col">
            <div class="chart-container">{_ks_curve_svg(y_true_oos, y_score_oos, width=400, height=280)}</div>
            <div class="chart-container">{_roc_curve_svg(y_true_oos, y_score_oos, width=400, height=280)}</div>
            <div class="chart-container">{_lift_chart_svg(y_true_oos, y_score_oos, width=400, height=280)}</div>
        </div>"""

        sections.append(("模型效果概览", charts_and_metrics))

    # ===== 3. 特征重要性 =====
    if feature_importance is not None and not feature_importance.empty:
        fi_html = _table_html(feature_importance, 'tbl_fi', show_all=True)
        # 条形图 Top 20
        top_fi = feature_importance.head(20) if hasattr(feature_importance, 'head') else feature_importance
        items = [(str(row.iloc[0]), float(row.iloc[1])) for _, row in top_fi.iterrows()]
        bars = f"<div class='subsection-title'>Top 20 特征重要性</div>{_bar_chart_html(items, 'neutral')}"
        sections.append(("特征重要性", bars + fi_html))

    # ===== 4. ModelDeliveryReport 全部报表 (Tab 导航) =====
    report_tables = {k: v for k, v in report_data.items()
                     if not k.startswith('_') and isinstance(v, pd.DataFrame) and not v.empty}

    if report_tables:
        tabs = []
        tab_contents = []
        for i, (name, df) in enumerate(report_tables.items()):
            tab_id = f"report_tab_{i}"
            tabs.append((tab_id, name))
            active = "tab-content active" if i == 0 else "tab-content"
            tab_contents.append(f'<div id="{tab_id}" class="{active}" data-tab-group="delivery_report">{_table_html(df, f"tbl_{tab_id}", show_all=True)}</div>')

        tab_nav = make_tab_nav(tabs, "delivery_report")
        tab_html = tab_nav + "".join(tab_contents)
        sections.append(("交付报告明细", tab_html))

    # 组装
    sections_html = ""
    for i, (s_title, content) in enumerate(sections):
        sections_html += f'<div class="section"><div class="section-title"><span class="badge">{i+1}</span> {s_title}</div>{content}</div>'

    html = _html_template(title, sections_html, now)
    if save_path:
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(html)
        print(f"✓ 模型评估 HTML 报告已保存: {save_path}")
    return html
