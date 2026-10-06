import streamlit as st
from typing import Callable, Any
from textwrap import dedent


# Streamlit parses four-space-indented HTML as a code block before it can
# honor unsafe_allow_html. Normalize such blocks once for every page that
# imports this shared UI module.
if not getattr(st.markdown, "_creditflow_html_dedent", False):
    _streamlit_markdown = st.markdown

    def _markdown_with_dedented_html(body, *args, **kwargs):
        if kwargs.get("unsafe_allow_html") and isinstance(body, str):
            body = dedent(body).strip()
        return _streamlit_markdown(body, *args, **kwargs)

    _markdown_with_dedented_html._creditflow_html_dedent = True
    st.markdown = _markdown_with_dedented_html


# ============================================================
# COLOR / BRAND TOKENS (Banking / AI palette)
# ============================================================

BRAND = {
    "primary": "#0b3d91",
    "primary_2": "#1e40af",
    "primary_light": "#dbeafe",
    "accent": "#0ea5e9",
    "accent_2": "#0284c7",
    "success": "#16a34a",
    "success_light": "#dcfce7",
    "warning": "#d97706",
    "warning_light": "#fef3c7",
    "danger":  "#dc2626",
    "danger_light": "#fee2e2",
    "ink": "#0f172a",
    "ink_2": "#1e293b",
    "muted": "#475569",
    "muted_2": "#64748b",
    "surface": "#ffffff",
    "bg_soft": "#f8fafc",
    "bg_gradient": (
        "linear-gradient(135deg, #eff6ff 0%, #ffffff 50%, #f0f9ff 100%);"
    ),
    "card_border": "#e2e8f0",
    "card_border_2": "#cbd5e1",
    "shadow": "0 1px 2px rgba(15, 23, 42, 0.04), 0 6px 18px rgba(15, 23, 42, 0.05);",
    "shadow_lg": "0 4px 8px rgba(15, 23, 42, 0.06), 0 16px 36px rgba(15, 23, 42, 0.08);"
}


# ============================================================
# GLOBAL CSS (injected once per page)
# ============================================================

UI_CSS = f"""
<style>

/* ====== Streamlit layout tweaks ====== */
.stMarkdown pre,
[data-testid="stCode"] {{
    display: none !important;
}}

.block-container {{
    padding-top: 1.4rem;
    padding-bottom: 3.8rem;
    max-width: 1440px;
}}

[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, #0b3d91 0%, #1e3a8a 55%, #0c4a6e 100%);
    color: #e2e8f0;
    border-right: 1px solid rgba(14,165,233,0.18);
    overflow: hidden;
}}
/* ====== NEW: Force Streamlit's giant duplicated sidebar header invisible ====== */
/* Targets both: the direct children above the first nav link, AND any headings
   rendered outside of stSidebarNav in the sidebar's top gutter area. */
section[data-testid="stSidebar"]
    > div[data-testid="stSidebarContent"]
    > div:first-child
    > *:not([data-testid="stSidebarNav"]):not([data-testid="stSidebarUserContent"]) {{
    display: none !important;
    visibility: hidden !important;
    height: 0 !important;
    opacity: 0 !important;
    pointer-events: none !important;
}}
/* Also cover the first ~100px of user content with a solid navy opaque strip
   so any leaked overlapping giant heading text is hidden behind our card area */
section[data-testid="stSidebar"]
    > div[data-testid="stSidebarContent"]
    > div:first-child
    > [data-testid="stSidebarUserContent"]::before {{
    content: "";
    display: block;
    height: 72px;
    margin-top: -72px;
    background: linear-gradient(180deg, #0b3d91 0%, #1e3a8a 100%);
    pointer-events: none;
}}
[data-testid="stSidebarNav"] {{
    padding-top: 8px;
    position: relative;
    z-index: 2;
}}
[data-testid="stSidebarNav"] [data-testid="stSidebarNavItems"] {{
    padding: 6px 8px 10px;
    gap: 5px;
}}
[data-testid="stSidebarNav"] a {{
    border: 1px solid transparent;
    border-radius: 10px;
    margin: 2px 0;
    padding: 8px 10px;
    transition: background .15s ease, border-color .15s ease, transform .15s ease;
}}
[data-testid="stSidebarNav"] a:hover {{
    background: rgba(255,255,255,.12);
    border-color: rgba(255,255,255,.18);
    transform: translateX(2px);
}}
[data-testid="stSidebarNav"] a[aria-current="page"] {{
    background: linear-gradient(105deg, rgba(56,189,248,.24), rgba(255,255,255,.12));
    border-color: rgba(125,211,252,.4);
    box-shadow: inset 3px 0 #38bdf8;
}}
[data-testid="stSidebarNav"] a p {{
    font-weight: 650;
    font-size: 13px;
}}
/* Hide the built-in duplicated app label ("CreditFlow AI") rendered above nav links in newer Streamlit */
[data-testid="stSidebarNav"] [data-testid="stSidebarNavSeparator"] + div,
[data-testid="stSidebarNav"] > div:first-child > div:first-child,
[data-testid="stSidebarNav"] [class*="navAppTitle"],
[data-testid="stSidebarNav"] [data-testid="stLogo"] + div,
section[data-testid="stSidebar"]
    [data-testid="stSidebarNav"]
    > div:first-of-type
    > div:not([data-testid="stSidebarNavItems"]):not([data-testid="stSidebarNavLink"]) {{
    display: none !important;
}}
/* Hide any oversized, bold, white headings ANYWHERE in the sidebar section
   (these are the duplicated nav labels overlapping our brand card). */
section[data-testid="stSidebar"] [style*="font-weight: 800"][style*="font-size: 36"],
section[data-testid="stSidebar"] [style*="font-weight:800"][style*="font-size:36"],
section[data-testid="stSidebar"] [style*="font-weight: 800"][style*="font-size: 2.25"],
section[data-testid="stSidebar"] [style*="font-weight:800"][style*="font-size:2.25"],
section[data-testid="stSidebar"] [style*="font-weight: 800"][style*="font-size: 2rem"],
section[data-testid="stSidebar"] [style*="font-weight:800"][style*="font-size:2rem"],
section[data-testid="stSidebar"] [style*="font-size: 36px"][style*="font-weight: 800"],
section[data-testid="stSidebar"] [style*="font-size: 2.25rem"][style*="font-weight:800"] {{
    display: none !important;
    visibility: hidden !important;
    height: 0 !important;
    opacity: 0 !important;
}}
/* Push sidebar user-content (our custom brand card) FAR enough below the nav
   that even the tallest nav header cannot overlap it. */
[data-testid="stSidebarUserContent"] {{
    margin-top: 20px !important;
    padding-top: 0 !important;
    position: relative;
    z-index: 5;
}}
[data-testid="stSidebarNav"] ul + div,
[data-testid="stSidebarNav"] [data-testid="stSidebarHeader"] {{
    margin-top: 0 !important;
}}
section[data-testid="stSidebar"] > div[data-testid="stSidebarContent"] {{
    padding-top: 6px !important;
}}
[data-testid="stSidebar"] * {{ color: #e2e8f0; }}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] small,
[data-testid="stSidebar"] span {{
    color: #e2e8f0 !important;
}}

[data-testid="stSidebar"] hr, [data-testid="stSidebar"] .stDivider > div {{
    border-color: rgba(255,255,255,0.15) !important;
}}

[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {{
    opacity: .82;
}}

[data-testid="stSidebar"] button {{
    background: rgba(255,255,255,0.08) !important;
    border: 1px solid rgba(255,255,255,0.22) !important;
    color: #fff !important;
}}

/* ====== Typography ====== */
html, body, [class*="css"] {{
    font-family: "Inter", "Segoe UI", -apple-system, BlinkMacSystemFont, Roboto, "Helvetica Neue", sans-serif;
    color: {BRAND['ink']};
    background: {BRAND['bg_soft']};
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
}}

h1 {{ font-weight: 800; letter-spacing: -0.02em; color: {BRAND['ink']}; }}
h2 {{ font-weight: 750; letter-spacing: -0.015em; color: {BRAND['ink']}; }}
h3 {{ font-weight: 700; color: {BRAND['ink_2']}; }}
h4 {{ font-weight: 650; color: #334155; }}
h5 {{ font-weight: 600; color: #475569; }}

p, li, span, label {{ color: #334155; line-height: 1.55; }}

a {{ color: {BRAND['primary_2']}; font-weight: 500; text-decoration: none; }}
a:hover {{ text-decoration: underline; }}

/* ====== Metric cards ====== */
[data-testid="stMetric"] {{
    background: linear-gradient(180deg, #ffffff 0%, #f8fbff 100%);
    border: 1px solid {BRAND['card_border']};
    border-radius: 16px;
    padding: 16px 18px;
    box-shadow: {BRAND['shadow']};
    transition: transform .12s ease, box-shadow .15s ease;
}}
[data-testid="stMetric"]:hover {{
    transform: translateY(-2px);
    box-shadow: {BRAND['shadow_lg']};
}}
[data-testid="stMetric"] label {{ font-weight: 650; color: {BRAND['muted']}; font-size: 12.5px; letter-spacing: .2px; }}
[data-testid="stMetricValue"] div {{ font-weight: 800 !important; color: {BRAND['ink']} !important; letter-spacing: -.01em; }}
[data-testid="stMetricDelta"] svg {{ display: none; }}

/* ====== Cards / sections ====== */
.pro-card {{
    background: {BRAND['surface']};
    border: 1px solid {BRAND['card_border']};
    border-radius: 16px;
    padding: 20px 22px;
    box-shadow: {BRAND['shadow']};
    margin-bottom: 16px;
    transition: box-shadow .15s ease, transform .12s ease;
}}
.pro-card:hover {{ box-shadow: {BRAND['shadow_lg']}; }}

.section-title {{
    display: flex;
    align-items: center;
    gap: 10px;
    font-size: 16.5px;
    font-weight: 800;
    color: {BRAND['ink']};
    margin: 6px 0 16px 0;
    padding-bottom: 10px;
    border-bottom: 2px solid #e2e8f0;
    letter-spacing: 0.1px;
    position: relative;
}}
.section-title::before {{
    content: "";
    display: inline-block;
    width: 5px; height: 20px;
    background: linear-gradient(180deg, {BRAND['primary']} 0%, {BRAND['accent']} 100%);
    border-radius: 3px;
    position: relative;
    top: -1px;
}}
.section-title::after {{
    content: "";
    position: absolute;
    bottom: -2px; left: 0;
    width: 60px; height: 2px;
    background: linear-gradient(90deg, {BRAND['primary']} 0%, {BRAND['accent']} 100%);
    border-radius: 2px;
}}

/* ====== Prediction verdict banners ====== */
.verdict-approved {{
    position: relative;
    background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%);
    border: 1.5px solid #16a34a;
    color: #064e3b;
    border-radius: 18px;
    padding: 26px 28px;
    box-shadow: 0 6px 24px rgba(22, 163, 74, 0.14), 0 2px 6px rgba(22, 163, 74, 0.06);
    margin: 16px 0;
    overflow: hidden;
}}
.verdict-approved::before {{
    content: "";
    position: absolute;
    top: 0; left: 0;
    width: 6px; height: 100%;
    background: #16a34a;
}}
.verdict-rejected {{
    position: relative;
    background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%);
    border: 1.5px solid #dc2626;
    color: #7f1d1d;
    border-radius: 18px;
    padding: 26px 28px;
    box-shadow: 0 6px 24px rgba(220, 38, 38, 0.14), 0 2px 6px rgba(220, 38, 38, 0.06);
    margin: 16px 0;
    overflow: hidden;
}}
.verdict-rejected::before {{
    content: "";
    position: absolute;
    top: 0; left: 0;
    width: 6px; height: 100%;
    background: #dc2626;
}}
.verdict-tag {{
    display: inline-block;
    padding: 5px 14px;
    border-radius: 999px;
    font-size: 12px;
    font-weight: 800;
    letter-spacing: .4px;
    margin-right: 10px;
    text-transform: uppercase;
}}
.verdict-approved .verdict-tag {{ background: #16a34a; color: #fff; box-shadow: 0 2px 6px rgba(22,163,74,.25); }}
.verdict-rejected .verdict-tag {{ background: #dc2626; color: #fff; box-shadow: 0 2px 6px rgba(220,38,38,.25); }}

.verdict-headline {{
    font-size: 22px;
    font-weight: 850;
    margin: 4px 0 8px 0;
    letter-spacing: -.015em;
    line-height: 1.2;
}}
.verdict-sub {{
    font-size: 14px;
    opacity: .92;
    line-height: 1.55;
}}
.verdict-prob-row {{
    display: flex;
    gap: 14px;
    margin-top: 14px;
    flex-wrap: wrap;
}}
.verdict-prob-chip {{
    background: rgba(255,255,255,.65);
    backdrop-filter: blur(6px);
    border-radius: 12px;
    padding: 10px 14px;
    border: 1px solid rgba(255,255,255,.8);
    flex: 1;
    min-width: 140px;
}}
.verdict-prob-chip .p-lbl {{ font-size: 11px; font-weight: 700; letter-spacing: .3px; text-transform: uppercase; opacity: .75; }}
.verdict-prob-chip .p-val {{ font-size: 20px; font-weight: 850; margin-top: 2px; letter-spacing: -.01em; }}
.verdict-prob-chip .p-sub {{ font-size: 12px; opacity: .7; margin-top: 2px; }}

/* ====== KPI / stat mini cards ====== */
.mini-stat {{
    border: 1px solid {BRAND['card_border']};
    border-radius: 14px;
    padding: 12px 14px;
    background: #fff;
    box-shadow: {BRAND['shadow']};
    transition: transform .12s ease, box-shadow .15s ease;
}}
.mini-stat:hover {{ transform: translateY(-1px); box-shadow: {BRAND['shadow_lg']}; }}
.mini-stat .lbl {{ color: {BRAND['muted']}; font-size: 11.5px; font-weight: 700; letter-spacing: .35px; text-transform: uppercase; }}
.mini-stat .val {{ color: {BRAND['ink']}; font-size: 22px; font-weight: 800; margin-top: 2px; letter-spacing: -.01em; }}
.mini-stat .sub {{ color: #64748b; font-size: 12px; margin-top: 3px; line-height: 1.4; }}

.mini-stat.success {{ border-color: #bbf7d0; background: linear-gradient(180deg, #ffffff 0%, #f0fdf4 100%); }}
.mini-stat.success .val {{ color: {BRAND['success']}; }}
.mini-stat.danger  {{ border-color: #fecaca; background: linear-gradient(180deg, #ffffff 0%, #fef2f2 100%); }}
.mini-stat.danger  .val {{ color: {BRAND['danger']}; }}
.mini-stat.warning {{ border-color: #fde68a; background: linear-gradient(180deg, #ffffff 0%, #fffbeb 100%); }}
.mini-stat.warning .val {{ color: {BRAND['warning']}; }}
.mini-stat.primary {{ border-color: #1e3a8a; background: linear-gradient(135deg, #1e40af 0%, #0b3d91 100%); }}
.mini-stat.primary .lbl,
.mini-stat.primary .val,
.mini-stat.primary .sub {{ color: #fff !important; font-weight: 700 !important; }}
.mini-stat.info    {{ border-color: #0369a1; background: linear-gradient(135deg, #0284c7 0%, #0ea5e9 100%); }}
.mini-stat.info .lbl,
.mini-stat.info .val,
.mini-stat.info .sub {{ color: #fff !important; font-weight: 700 !important; }}

/* ====== Buttons ====== */
.stButton > button, .stDownloadButton > button, .stFormSubmitButton > button {{
    background: linear-gradient(180deg, {BRAND['primary']} 0%, {BRAND['primary_2']} 100%);
    color: #fff !important;
    font-weight: 700 !important;
    border-radius: 10px;
    border: 1px solid #0b3d91;
    padding: .6rem 1.05rem;
    transition: transform .08s ease, box-shadow .18s ease, filter .15s ease;
    box-shadow: 0 2px 6px rgba(11, 61, 145, 0.2);
    letter-spacing: .2px;
}}
.stButton > button:hover, .stDownloadButton > button:hover, .stFormSubmitButton > button:hover {{
    filter: brightness(1.06);
    transform: translateY(-1px);
    box-shadow: 0 6px 16px rgba(11, 61, 145, 0.28);
}}
.stButton > button:active, .stDownloadButton > button:active, .stFormSubmitButton > button:active {{
    transform: translateY(0);
    filter: brightness(.98);
}}

/* ====== Select boxes, sliders, inputs ====== */
.stSlider [data-testid="stTickBar"] * {{ color: #475569; }}
[data-baseweb="select"] > div,
[data-baseweb="input"] > div,
[data-baseweb="base-input"] > div,
.stTextArea textarea,
.stSelectbox [data-baseweb="select"] {{
    border-radius: 10px !important;
    border-color: {BRAND['card_border_2']} !important;
    background: #fff !important;
    transition: border-color .15s ease, box-shadow .15s ease;
}}
[data-baseweb="select"] > div:focus-within,
[data-baseweb="input"] > div:focus-within,
.stTextArea textarea:focus {{
    border-color: {BRAND['primary']} !important;
    box-shadow: 0 0 0 3px rgba(11,61,145,.10);
}}

div[data-testid="stForm"] {{
    border: 1px solid {BRAND['card_border']};
    border-radius: 16px;
    padding: 22px 24px;
    background: linear-gradient(180deg, #ffffff 0%, #fafcff 100%);
    box-shadow: {BRAND['shadow']};
}}

/* ====== Dataframes / tables ====== */
[data-testid="stDataFrame"] {{
    border-radius: 14px;
    border: 1px solid {BRAND['card_border']};
    overflow: hidden;
    box-shadow: {BRAND['shadow']};
}}

/* ====== Charts ====== */
[data-testid="stScatterChart"],
[data-testid="stBarChart"],
[data-testid="stLineChart"],
[data-testid="stVegaLiteChart"] {{
    border-radius: 14px;
    border: 1px solid {BRAND['card_border']};
    overflow: hidden;
    background: #fff;
    box-shadow: {BRAND['shadow']};
    padding: 6px;
}}

/* ====== Tab styling ====== */
.stTabs [data-baseweb="tab-list"] {{
    gap: 4px;
    background: #f1f5f9;
    border-radius: 12px;
    padding: 6px;
    margin-bottom: 14px;
}}
.stTabs [data-baseweb="tab"] {{
    border-radius: 8px !important;
    padding: 8px 16px !important;
    font-weight: 650 !important;
    letter-spacing: .2px;
    color: #475569;
    background: transparent;
    transition: background .15s ease, color .15s ease;
}}
.stTabs [data-baseweb="tab"]:hover {{
    background: rgba(255,255,255,.6);
    color: {BRAND['ink_2']};
}}
.stTabs [data-baseweb="tab"][aria-selected="true"] {{
    background: #fff;
    color: {BRAND['primary_2']};
    box-shadow: 0 2px 6px rgba(15,23,42,.08);
}}
.stTabs [data-baseweb="tab-highlight"] {{
    display: none !important;
}}

/* ====== Expander ====== */
[data-testid="stExpander"] {{
    border: 1px solid {BRAND['card_border']} !important;
    border-radius: 14px !important;
    background: #fff;
    box-shadow: {BRAND['shadow']};
    overflow: hidden;
}}
[data-testid="stExpander"] details summary p {{
    font-weight: 700;
    color: {BRAND['ink_2']};
}}

/* ====== Probability gauge / bar ====== */
.prob-bar-wrap {{
    background: #f1f5f9;
    border-radius: 999px;
    height: 18px;
    overflow: hidden;
    position: relative;
    border: 1px solid {BRAND['card_border']};
}}
.prob-bar-fill {{
    height: 100%;
    border-radius: 999px;
    transition: width .6s cubic-bezier(.4,0,.2,1);
    background: linear-gradient(90deg, {BRAND['danger']} 0%, {BRAND['warning']} 45%, {BRAND['success']} 100%);
    position: relative;
}}
.prob-bar-fill::after {{
    content: "";
    position: absolute;
    top: 0; right: 0; bottom: 0;
    width: 40%;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,.35));
    border-radius: 999px;
}}
.prob-marker {{
    position: absolute;
    top: -5px;
    width: 3px;
    height: 28px;
    background: {BRAND['ink']};
    border-radius: 2px;
    transform: translateX(-50%);
    z-index: 2;
    box-shadow: 0 0 0 3px rgba(15,23,42,.08);
}}
.prob-legend {{
    display: flex;
    justify-content: space-between;
    font-size: 11.5px;
    font-weight: 650;
    color: {BRAND['muted']};
    margin-top: 6px;
    letter-spacing: .2px;
}}

/* ====== Risk badges ====== */
.risk-badge {{
    display: inline-block;
    padding: 4px 12px;
    border-radius: 999px;
    font-size: 12px;
    font-weight: 800;
    letter-spacing: .25px;
    text-transform: uppercase;
    border: 1px solid;
}}
.risk-low    {{ background: {BRAND['success_light']}; color: #166534; border-color: #86efac; }}
.risk-medium {{ background: {BRAND['warning_light']}; color: #92400e; border-color: #fcd34d; }}
.risk-high   {{ background: {BRAND['danger_light']};  color: #991b1b; border-color: #fca5a5; }}
.risk-info   {{ background: linear-gradient(135deg, #1e40af 0%, #1e3a8a 100%); color: #fff; border-color: #0b3d91; font-weight: 800; }}

/* ====== Summary card ====== */
.summary-card {{
    border: 1px solid {BRAND['card_border']};
    border-radius: 14px;
    padding: 18px 20px;
    background: linear-gradient(135deg, #ffffff 0%, #f6f8ff 100%);
    box-shadow: {BRAND['shadow']};
}}
.summary-row {{
    display: flex;
    justify-content: space-between;
    padding: 7px 0;
    border-bottom: 1px dashed #e2e8f0;
    font-size: 13.5px;
    align-items: center;
}}
.summary-row:last-child {{ border-bottom: none; }}
.summary-label {{ color: #64748b; font-weight: 550; }}
.summary-value {{ color: {BRAND['ink_2']}; font-weight: 700; }}
.summary-row.highlight {{
    background: linear-gradient(135deg, #1e40af 0%, #0b3d91 100%);
    border-radius: 8px;
    padding: 9px 12px;
    margin: 4px -12px;
    border-bottom: none;
}}
.summary-row.highlight .summary-label,
.summary-row.highlight .summary-value {{
    color: #fff !important;
    font-weight: 700 !important;
}}

/* ====== Page hero header ====== */
.page-hero {{
    border: 1px solid {BRAND['card_border']};
    border-radius: 20px;
    padding: 22px 26px;
    background: linear-gradient(135deg, #eff6ff 0%, #ffffff 45%, #ecfeff 100%);
    box-shadow: 0 10px 30px rgba(2,132,199,0.08);
    margin-bottom: 22px;
    position: relative;
    overflow: hidden;
}}
.page-hero::after {{
    content: "";
    position: absolute;
    top: -40px; right: -40px;
    width: 200px; height: 200px;
    background: radial-gradient(circle, rgba(14,165,233,.08) 0%, transparent 70%);
    border-radius: 50%;
}}
.hero-chip {{
    display: inline-block;
    padding: 5px 14px;
    border-radius: 999px;
    background: linear-gradient(135deg, #1e40af 0%, #0b3d91 100%);
    color: #fff;
    font-size: 12px;
    font-weight: 800;
    letter-spacing: .35px;
    margin-bottom: 10px;
    text-transform: uppercase;
    border: 1px solid rgba(255,255,255,0.2);
}}
.hero-title {{
    font-size: 28px;
    font-weight: 850;
    letter-spacing: -.022em;
    color: {BRAND['ink']};
    margin: 4px 0 8px 0;
    line-height: 1.15;
}}
.hero-sub {{
    color: #334155;
    line-height: 1.65;
    font-size: 14.5px;
}}

/* ====== Responsive helper ====== */
@media (max-width: 768px) {{
    .pro-card, .summary-card {{ padding: 14px 16px; }}
    h1, .hero-title {{ font-size: 24px !important; }}
    .page-hero {{ padding: 16px 18px; }}
    .block-container {{ padding-top: 1rem; }}
}}

/* ====== Footer / disclaimer ====== */
.app-footer {{
    margin-top: 40px;
    padding: 20px 22px;
    border: 1px solid #fde68a;
    background: linear-gradient(135deg, #fffbeb 0%, #ffffff 100%);
    border-radius: 16px;
    color: #78350f;
    box-shadow: {BRAND['shadow']};
    position: relative;
    overflow: hidden;
}}
.app-footer::before {{
    content: "⚠";
    position: absolute;
    top: 14px; right: 18px;
    font-size: 38px;
    opacity: .08;
    font-weight: 900;
}}
.app-footer .title {{
    font-weight: 850;
    color: #92400e;
    margin-bottom: 8px;
    font-size: 15px;
    letter-spacing: .2px;
}}
.app-footer p, .app-footer li, .app-footer span {{ color: #78350f; line-height: 1.6; }}
.app-footer ul {{ margin: 6px 0 0 0; padding-left: 20px; }}

/* ====== Loading skeleton ====== */
.skeleton {{
    background: linear-gradient(90deg, #f1f5f9 0%, #e2e8f0 50%, #f1f5f9 100%);
    background-size: 200% 100%;
    animation: shimmer 1.4s ease-in-out infinite;
    border-radius: 8px;
}}
@keyframes shimmer {{
    0%   {{ background-position: 200% 0; }}
    100% {{ background-position: -200% 0; }}
}}

/* ====== Alert / info boxes ====== */
.alert-box {{
    border-radius: 14px;
    padding: 14px 18px;
    margin: 10px 0;
    border: 1px solid;
    font-size: 14px;
    line-height: 1.55;
}}
.alert-box.info     {{ background: linear-gradient(135deg, #1e40af 0%, #0b3d91 100%); border-color: #1e3a8a; color: #fff; font-weight: 700; }}
.alert-box.info *   {{ color: #fff !important; font-weight: 700 !important; }}
.alert-box.success  {{ background: #f0fdf4; border-color: #bbf7d0; color: #14532d; }}
.alert-box.warning  {{ background: #fffbeb; border-color: #fde68a; color: #78350f; }}
.alert-box.danger   {{ background: #fef2f2; border-color: #fecaca; color: #7f1d1d; }}

/* Streamlit native st.info() override */
[data-testid="stAlert"] {{
    background: linear-gradient(135deg, #1e40af 0%, #0b3d91 100%) !important;
    border: 1px solid #1e3a8a !important;
    border-radius: 14px !important;
}}
[data-testid="stAlert"] * {{
    color: #fff !important;
    font-weight: 700 !important;
}}

/* ====== Record card (for history) ====== */
.record-card {{
    border: 1px solid {BRAND['card_border']};
    border-radius: 14px;
    padding: 16px 18px;
    margin-bottom: 12px;
    background: linear-gradient(135deg,#fff 0%,#f8faff 100%);
    box-shadow: {BRAND['shadow']};
    transition: transform .12s ease, box-shadow .15s ease;
}}
.record-card:hover {{ transform: translateY(-1px); box-shadow: {BRAND['shadow_lg']}; }}

.badge {{
    display: inline-block;
    padding: 3px 10px;
    border-radius: 999px;
    font-size: 11.5px;
    font-weight: 750;
    letter-spacing: .25px;
    border: 1px solid;
}}
.badge-yes {{ background: #dcfce7; color: #166534; border-color: #86efac; }}
.badge-no  {{ background: #fee2e2; color: #991b1b; border-color: #fca5a5; }}
.badge-ch  {{ background: linear-gradient(135deg, #3730a3 0%, #1e40af 100%); color: #fff; border-color: #312e81; font-weight: 800; }}
.badge-mdl {{ background: #fff7e6; color: #92400e; border-color: #fde68a; }}

.kv {{
    display:flex;justify-content:space-between;
    padding:5px 0;font-size:13.5px;
}}
.kv .k {{ color:#4b5563; font-weight: 500; }}
.kv .v {{ font-weight: 700; color: #111827; }}

/* ====== Feature card (XAI) ====== */
.feature-card {{
    border:1px solid {BRAND['card_border']};
    border-radius:12px;
    padding:14px 16px;
    background:#ffffff;
    margin-bottom:10px;
    box-shadow: {BRAND['shadow']};
    transition: transform .12s ease;
}}
.feature-card:hover {{ transform: translateY(-1px); }}
.pos-contrib {{
    color: #166534;
    font-weight: 700;
    background: #dcfce7;
    padding: 2px 8px;
    border-radius: 6px;
    font-size: 12px;
}}
.neg-contrib {{
    color: #991b1b;
    font-weight: 700;
    background: #fee2e2;
    padding: 2px 8px;
    border-radius: 6px;
    font-size: 12px;
}}

/* ====== Stat card variants ====== */
.stat-card {{
    border:1px solid {BRAND['card_border']};
    border-radius:12px;
    padding:14px 16px;
    background:linear-gradient(135deg,#fff 0%,#f6f8ff 100%);
    box-shadow: {BRAND['shadow']};
}}
.stat-head {{
    font-size:12px;color:#6b7280;text-transform:uppercase;
    letter-spacing:.35px;font-weight:700;margin-bottom:4px;
}}
.stat-val {{
    font-size:20px;font-weight:800;color:#1a365d;letter-spacing:-.01em;
}}
.stat-sub {{
    font-size:12px;color:#4b5563;margin-top:3px;line-height:1.4;
}}

/* ====== Chat bubble styles ====== */
.chat-user {{
    background: linear-gradient(135deg, {BRAND['primary']} 0%, {BRAND['primary_2']} 100%);
    color: #fff;
    padding: 10px 14px;
    border-radius: 14px 14px 4px 14px;
    margin: 6px 0;
    box-shadow: 0 2px 6px rgba(11,61,145,.18);
    max-width: 85%;
    margin-left: auto;
}}
.chat-user * {{ color: #fff !important; }}
.chat-bot {{
    background: #fff;
    border: 1px solid {BRAND['card_border']};
    padding: 10px 14px;
    border-radius: 14px 14px 14px 4px;
    margin: 6px 0;
    box-shadow: {BRAND['shadow']};
    max-width: 85%;
}}
.chip {{
    display: inline-block;
    background: linear-gradient(135deg, #1e40af 0%, #0b3d91 100%);
    color: #fff;
    padding: 6px 12px;
    border-radius: 999px;
    font-size: 12.5px;
    font-weight: 700;
    margin: 3px 4px 3px 0;
    border: 1px solid rgba(255,255,255,0.2);
    cursor: pointer;
    transition: all .15s ease;
}}
.chip:hover {{
    background: linear-gradient(135deg, #0b3d91 0%, #1e40af 100%);
    transform: translateY(-1px);
}}

/* ====== Shared: BASE / TUNED model badges ====== */
.base-badge {{
    display: inline-block;
    padding: 3px 12px;
    border-radius: 999px;
    font-size: 11.5px;
    letter-spacing: .3px;
    font-weight: 800 !important;
    text-transform: uppercase;
    border: 1px solid #cbd5e1;
    background: #f8fafc;
    color: #334155;
}}
.tuned-badge {{
    display: inline-block;
    padding: 3px 12px;
    border-radius: 999px;
    font-size: 11.5px;
    letter-spacing: .3px;
    font-weight: 800 !important;
    text-transform: uppercase;
    border: 1px solid #1e3a8a;
    background: linear-gradient(135deg, #1e40af 0%, #0b3d91 100%);
    color: #fff !important;
    box-shadow: 0 2px 6px rgba(11,61,145,0.28);
}}

/* ====== Confusion-matrix cell (cm-cell) ====== */
.cm-cell {{
    border-radius: 10px;
    padding: 10px 12px;
    font-weight: 800;
    text-align: center;
    font-size: 13px;
}}

/* ====== XAI / SHAP feature cards + text helpers ====== */
.feature-headline {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 10px;
    margin-bottom: 4px;
}}
.feature-name {{
    font-weight: 750;
    color: #0f172a;
    font-size: 13.5px;
}}
.contrib-value {{
    font-weight: 800;
    font-size: 13px;
}}
.pos-text {{
    color: #166534;
    font-weight: 800;
    background: #dcfce7;
    padding: 2px 8px;
    border-radius: 6px;
    font-size: 12px;
    display: inline-block;
}}
.neg-text {{
    color: #991b1b;
    font-weight: 800;
    background: #fee2e2;
    padding: 2px 8px;
    border-radius: 6px;
    font-size: 12px;
    display: inline-block;
}}
.rank-badge {{
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 24px; height: 24px;
    border-radius: 8px;
    font-weight: 800;
    font-size: 12px;
    color: #fff !important;
    background: linear-gradient(135deg, #1e40af 0%, #0b3d91 100%);
    margin-right: 8px;
}}

/* Keep text readable and consistent in every blue UI box. */
.mini-stat.primary,
.mini-stat.info,
.summary-row.highlight,
.alert-box.info,
[data-testid="stAlert"],
.risk-info,
.hero-chip,
.chat-user,
.chip,
.tuned-badge,
.badge-ch,
.rank-badge {{
    color: #fff !important;
    font-weight: 700 !important;
}}
.mini-stat.primary *,
.mini-stat.info *,
.summary-row.highlight *,
.alert-box.info *,
[data-testid="stAlert"] *,
.risk-info *,
.hero-chip *,
.chat-user *,
.chip *,
.tuned-badge *,
.badge-ch *,
.rank-badge * {{
    color: #fff !important;
    font-weight: 700 !important;
}}
.stButton > button *,
.stDownloadButton > button *,
.stFormSubmitButton > button *,
[data-testid="stSidebar"] button * {{
    color: #fff !important;
    font-weight: 700 !important;
    -webkit-text-fill-color: #fff !important;
}}
/* Streamlit renders button labels inside nested markdown elements. */
div[data-testid="stButton"] button,
div[data-testid="stButton"] button p,
div[data-testid="stButton"] button span,
div[data-testid="stButton"] button div,
div[data-testid="stFormSubmitButton"] button,
div[data-testid="stFormSubmitButton"] button p,
div[data-testid="stFormSubmitButton"] button span,
div[data-testid="stDownloadButton"] button,
div[data-testid="stDownloadButton"] button p,
div[data-testid="stDownloadButton"] button span {{
    color: #fff !important;
    font-weight: 700 !important;
    -webkit-text-fill-color: #fff !important;
}}

</style>
"""


def inject_theme_css() -> None:
    """Inject the professional banking/AI theme + CSS once per page."""
    st.markdown(UI_CSS, unsafe_allow_html=True)


# ============================================================
# SIDEBAR BRANDING (shared)
# ============================================================

def render_sidebar_brand(app_name: str = "CreditFlow AI", subtitle: str = "Loan decision support") -> None:
    """Render a compact, consistent brand header above the page navigation."""
    app_name = "CreditFlow AI"
    subtitle = "Loan decision support"
    with st.sidebar:
        st.markdown(
            f"""
            <div style="
                border:1px solid rgba(255,255,255,0.25);
                background: linear-gradient(135deg, rgba(14,165,233,0.18), rgba(30,64,175,0.35));
                backdrop-filter: blur(6px);
                border-radius: 14px;
                padding: 10px 12px;
                margin-top: 8px;
                margin-bottom: 8px;
                position: relative;
                z-index: 9999;
                box-shadow: 0 4px 14px rgba(0,0,0,0.18);
            ">
                <div style="display:flex;align-items:center;gap:10px;">
                    <div style="
                        flex: 0 0 auto;
                        width:40px;height:40px;border-radius:10px;
                        background: linear-gradient(135deg, #38bdf8 0%, #1d4ed8 100%);
                        display:flex;align-items:center;justify-content:center;
                        font-size:20px;font-weight:900;color:#fff;
                        box-shadow:0 4px 12px rgba(14,165,233,.45);
                    ">🏦</div>
                    <div style="min-width:0;flex:1;">
                <div style="font-size:15px;font-weight:800;line-height:1.15;color:#fff !important;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">
                            {app_name}
                        </div>
                        <div style="font-size:11px;color:#bae6fd !important;margin-top:3px;line-height:1.25;">
                            {subtitle}
                        </div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


# ============================================================
# PROMINENT PREDICTION BANNER
# ============================================================

def render_verdict_banner(
    approved: bool,
    probability: float,
    model_name: str,
    subtitle: str = ""
) -> None:
    """Render a large Approved / Rejected verdict banner like a real core-banking UI."""

    cls = "verdict-approved" if approved else "verdict-rejected"
    tag_text = "✓ LOAN APPROVED" if approved else "✗ LOAN REJECTED"
    verdict_title = "Recommended Decision: **APPROVE**" if approved else "Recommended Decision: **REJECT**"

    st.markdown(
        f"""
        <div class="{cls}">
            <div style="display:flex;align-items:center;flex-wrap:wrap;gap:8px;margin-bottom:6px;">
                <span class="verdict-tag">{tag_text}</span>
                <span style="opacity:.9;font-weight:600;font-size:13px;">
                    Model: {model_name}
                </span>
            </div>
            <div style="font-size:20px;font-weight:800;line-height:1.25;margin:2px 0 6px 0;">
                {verdict_title}
            </div>
            <div style="font-size:14px;opacity:.92;">
                <b>Approval Probability:</b> {probability*100:.2f}%
                (decision threshold 50%)
            </div>
            {f'<div style="font-size:13px;opacity:.82;margin-top:6px;">' + subtitle + '</div>' if subtitle else ''}
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# MINI STAT CARD (text, no JS/HTML)
# ============================================================

def mini_stat_html(label: str, value: str, sub: str = "", width: int = 100) -> str:
    """Return HTML string for a mini stat card. Embed with st.markdown(unsafe_allow_html=True)."""
    w = f"width:{width}%;" if width != 100 else "width:100%;"
    return f"""
        <div class="mini-stat" style="{w}">
            <div class="lbl">{label}</div>
            <div class="val">{value}</div>
            {f'<div class="sub">{sub}</div>' if sub else ''}
        </div>
    """


# ============================================================
# SECTION TITLE HELPER
# ============================================================

def section_title(title: str) -> None:
    """Render a consistent, themed section title."""
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)


# ============================================================
# PRO CARD WRAPPER (with st.container)
# ============================================================

def pro_card(contents: Callable[[], Any]) -> None:
    """Render callable content inside a themed professional card container."""
    st.markdown('<div class="pro-card">', unsafe_allow_html=True)
    contents()
    st.markdown('</div>', unsafe_allow_html=True)


# ============================================================
# ACADEMIC DISCLAIMER FOOTER (shared)
# ============================================================

DISCLAIMER_TITLE = "⚠️ Academic-Use & Decision-Support Disclaimer"

DISCLAIMER_BODY = """
**CreditFlow AI – Loan Approval DSS** is an **academic decision-support prototype** only.

- This tool **does not constitute a formal credit offer, commitment, or legal lending decision**.
- All predictions are generated by Machine Learning models trained on a **single historical dataset** (loan_data.csv) and are subject to **dataset bias, concept drift, temporal drift, class imbalance, and sampling limitations**.
- Every lending decision must be reviewed manually by a **licensed / authorised underwriter** in accordance with applicable regulations (e.g. fair lending, data-privacy rules) before any offer is made to an applicant.
- Probability scores, SHAP explanations, metrics, and rankings are **interpretive aids** only and are not guaranteed.
- Do not use this tool to make real-world credit decisions, reject customers, or price risk in a production environment without a full model-validation, audit, and regulatory review.
- The UI is branded for demonstration purposes only; no banking or financial institution is affiliated.
"""

def render_disclaimer_footer() -> None:
    """Render the shared academic disclaimer footer card."""
    st.markdown(
        f"""
        <div class="app-footer">
            <div class="title">{DISCLAIMER_TITLE}</div>
            <div style="font-size:13px;line-height:1.6;">
                {DISCLAIMER_BODY.replace(chr(10), '<br>')}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# LOADING / ERROR WRAPPERS
# ============================================================

def loading_spinner(message: str = "Loading …"):
    """Return a streamlit spinner with consistent messaging."""
    return st.spinner(f"⏳ {message}")


def render_error_state(message: str, hint: str = "") -> None:
    """Render a consistent, styled error state with optional hint."""
    err_body = f"**❌ Error:** {message}"
    if hint:
        err_body += f"\n\n💡 **Hint:** {hint}"
    st.error(err_body)


def render_empty_state(title: str, body: str, action_hint: str = "") -> None:
    """Render a consistent empty/placeholder info state (e.g. no predictions yet)."""
    st.info(f"**ℹ️ {title}**\n\n{body}" + (f"\n\n👉 {action_hint}" if action_hint else ""))


# ============================================================
# HEADER (for Dashboard home)
# ============================================================

def render_dashboard_header(
    dataset_rows: int,
    dataset_cols: int,
    approval_rate: float,
    n_models: int
) -> None:
    """Render the big dashboard hero header used on the Dashboard home page."""

    st.markdown(
        f"""
        <div class="page-hero">
            <div style="display:flex;align-items:flex-start;justify-content:space-between;gap:22px;flex-wrap:wrap;position:relative;z-index:1;">
                <div style="min-width:300px;">
                    <span class="hero-chip">Executive Dashboard · CreditFlow AI</span>
                    <div class="hero-title">Loan Approval Decision Support</div>
                    <div class="hero-sub">
                        Machine Learning models (Logistic, Tree, RF, SVM, XGB + tuned variants)
                        trained on <b>{dataset_rows:,}</b> historical applications across
                        <b>{dataset_cols}</b> features. Current dataset approval rate is
                        <b>{approval_rate:.1f}%</b>, with <b>{n_models}</b> trained classifiers
                        ready for single-run predictions, agreement analysis and SHAP
                        explainability.
                    </div>
                </div>
                <div style="display:grid;grid-template-columns:repeat(2,170px);gap:12px;">
                    <div class="mini-stat primary">
                        <div class="lbl">Applications</div>
                        <div class="val">{dataset_rows:,}</div>
                        <div class="sub">{dataset_cols} features</div>
                    </div>
                    <div class="mini-stat success">
                        <div class="lbl">Approval Rate</div>
                        <div class="val">{approval_rate:.1f}%</div>
                        <div class="sub">Historical log</div>
                    </div>
                    <div class="mini-stat info">
                        <div class="lbl">Models Ready</div>
                        <div class="val">{n_models}</div>
                        <div class="sub">Base + Tuned</div>
                    </div>
                    <div class="mini-stat">
                        <div class="lbl">Explainability</div>
                        <div class="val">SHAP</div>
                        <div class="sub">Global & local</div>
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# PAGE HERO HEADER (generic for sub-pages)
# ============================================================

def render_page_hero(
    chip_label: str,
    title: str,
    subtitle: str,
    mini_stats: list = None
) -> None:
    """Render a consistent page hero using native Streamlit components.

    mini_stats: optional list of dicts with keys [label, value, sub, variant]
                variant ∈ {success, danger, warning, primary, info, default}
    """
    with st.container(border=True):
        st.caption(chip_label.upper())
        st.title(title)
        st.write(subtitle)

        if mini_stats:
            stat_columns = st.columns(min(len(mini_stats), 4))
            for col, stat in zip(stat_columns, mini_stats):
                with col:
                    with st.container(border=True):
                        st.caption(str(stat.get("label", "")))
                        st.subheader(str(stat.get("value", "")))
                        if stat.get("sub"):
                            st.caption(str(stat["sub"]))


# ============================================================
# ENHANCED VERDICT BANNER (with probability chips + gauge)
# ============================================================

def render_verdict_enhanced(
    approved: bool,
    probability: float,
    model_name: str,
    total_income: float = None,
    loan_amount: float = None,
    lti: float = None,
    credit_good: bool = None,
    subtitle: str = ""
) -> None:
    """Enhanced verdict banner with probability chips + contextual stats."""

    cls = "verdict-approved" if approved else "verdict-rejected"
    tag_text = "✓ LOAN APPROVED" if approved else "✗ LOAN REJECTED"
    headline = "Recommended Decision: APPROVE" if approved else "Recommended Decision: REJECT"
    threshold_delta = (probability - 0.5) * 100
    delta_txt = f"{threshold_delta:+.1f} pp vs 50% threshold"
    rej_txt = f"{(1 - probability) * 100:.1f}% · {(1 - probability) * 100 - 50:+.1f} pp"

    extra_chips = ""
    if total_income is not None:
        extra_chips += f"""
            <div class="verdict-prob-chip">
                <div class="p-lbl">Total Income</div>
                <div class="p-val">{total_income:,.0f}</div>
                <div class="p-sub">Household combined</div>
            </div>
        """
    if loan_amount is not None:
        extra_chips += f"""
            <div class="verdict-prob-chip">
                <div class="p-lbl">Loan Amount</div>
                <div class="p-val">{loan_amount:,.0f}</div>
                <div class="p-sub">Principal requested</div>
            </div>
        """
    if lti is not None:
        risk_lbl = "LOW" if lti < 0.02 else ("MEDIUM" if lti < 0.04 else "HIGH")
        extra_chips += f"""
            <div class="verdict-prob-chip">
                <div class="p-lbl">LTI Ratio</div>
                <div class="p-val">{lti:.4f}</div>
                <div class="p-sub">Risk: {risk_lbl}</div>
            </div>
        """
    if credit_good is not None:
        credit_txt = "Meets criteria" if credit_good else "Needs review"
        extra_chips += f"""
            <div class="verdict-prob-chip">
                <div class="p-lbl">Credit History</div>
                <div class="p-val">{'✓' if credit_good else '✗'}</div>
                <div class="p-sub">{credit_txt}</div>
            </div>
        """

    st.markdown(
        f"""
        <div class="{cls}">
            <div style="display:flex;align-items:center;flex-wrap:wrap;gap:8px;margin-bottom:8px;">
                <span class="verdict-tag">{tag_text}</span>
                <span style="opacity:.9;font-weight:600;font-size:13px;">
                    Model: <b>{model_name}</b>
                </span>
            </div>
            <div class="verdict-headline">{headline}</div>
            <div class="verdict-sub">
                {subtitle if subtitle else 'Model prediction outcome based on applicant profile and training data.'}
            </div>
            <div class="verdict-prob-row">
                <div class="verdict-prob-chip">
                    <div class="p-lbl">Approval Probability</div>
                    <div class="p-val">{probability*100:.1f}%</div>
                    <div class="p-sub">{delta_txt}</div>
                </div>
                <div class="verdict-prob-chip">
                    <div class="p-lbl">Rejection Probability</div>
                    <div class="p-val">{(1-probability)*100:.1f}%</div>
                    <div class="p-sub">{rej_txt if not approved else ""}</div>
                </div>
                {extra_chips}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# PROBABILITY GAUGE / BAR (visualization)
# ============================================================

def render_probability_gauge(
    probability: float,
    threshold: float = 0.5,
    title: str = "Approval Probability Spectrum",
    show_threshold: bool = True
) -> None:
    """Render a gradient probability bar with threshold marker and legend."""

    pct = max(0.0, min(1.0, probability)) * 100
    threshold_pct = max(0.0, min(1.0, threshold)) * 100
    marker_html = f'<div class="prob-marker" style="left:{threshold_pct}%;"></div>' if show_threshold else ""

    st.markdown(
        f"""
        <div class="pro-card" style="padding:16px 18px;">
            <div style="display:flex;justify-content:space-between;align-items:flex-end;margin-bottom:10px;">
                <div style="font-weight:750;color:{BRAND['ink_2']};font-size:15px;">{title}</div>
                <div style="text-align:right;">
                    <div style="font-size:11.5px;color:{BRAND['muted']};font-weight:700;letter-spacing:.3px;text-transform:uppercase;">Score</div>
                    <div style="font-size:26px;font-weight:850;color:{BRAND['ink']};letter-spacing:-.02em;line-height:1;">{pct:.1f}%</div>
                </div>
            </div>
            <div style="position:relative;">
                <div class="prob-bar-wrap">
                    <div class="prob-bar-fill" style="width:{pct}%;"></div>
                    {marker_html}
                </div>
                <div class="prob-legend">
                    <span>0% · High Risk</span>
                    <span>50% · Threshold</span>
                    <span>100% · Low Risk</span>
                </div>
            </div>
            {f'<div style="margin-top:8px;font-size:12.5px;color:{BRAND["muted_2"]};line-height:1.5;">Threshold set to <b>{threshold*100:.0f}%</b>. Scores at or above trigger an <b>Approved</b> recommendation.</div>' if show_threshold else ""}
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# MINI STAT (rendered) + VARIANT
# ============================================================

def render_mini_stat(
    label: str,
    value: str,
    sub: str = "",
    variant: str = "default"
) -> None:
    """Render a themed mini-stat card directly into the app.

    variant ∈ {success, danger, warning, primary, info, default}
    """
    cls = f"mini-stat {variant}" if variant != "default" else "mini-stat"
    st.markdown(
        f"""
        <div class="{cls}">
            <div class="lbl">{label}</div>
            <div class="val">{value}</div>
            {f'<div class="sub">{sub}</div>' if sub else ""}
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# RISK BADGE HTML GENERATOR
# ============================================================

def risk_badge_html(level: str) -> str:
    """Return HTML for a risk badge. level ∈ {LOW, MEDIUM, HIGH, INFO}."""
    lvl = level.upper().strip()
    mapping = {
        "LOW": "risk-low",
        "MEDIUM": "risk-medium",
        "HIGH": "risk-high",
        "INFO": "risk-info",
    }
    cls = mapping.get(lvl, "risk-info")
    return f'<span class="risk-badge {cls}">{lvl}</span>'


def render_risk_badge(level: str) -> None:
    """Render a risk badge directly."""
    st.markdown(risk_badge_html(level), unsafe_allow_html=True)


# ============================================================
# ALERT BOX (rendered)
# ============================================================

def render_alert(message: str, kind: str = "info") -> None:
    """Render a styled alert box. kind ∈ {info, success, warning, danger}."""
    st.markdown(
        f'<div class="alert-box {kind}">{message}</div>',
        unsafe_allow_html=True
    )


# ============================================================
# PAGE INIT HELPER (boilerplate per page)
# ============================================================

def init_page(
    page_title: str,
    page_icon: str,
    sidebar_app_name: str,
    sidebar_subtitle: str,
    sidebar_description: str = ""
) -> None:
    """One-call page bootstrap: set_page_config + theme CSS + sidebar brand + optional description.

    Usage (top of any page file):
        init_page("Loan Prediction · CreditFlow AI", "🔮", "Loan Prediction",
                  "Run applicant approval checks", "Fill the form, get ML verdict.")
    """
    st.set_page_config(
        page_title=page_title,
        page_icon=page_icon,
        layout="wide",
        initial_sidebar_state="expanded"
    )
    inject_theme_css()
    render_sidebar_brand(sidebar_app_name, sidebar_subtitle)
    if sidebar_description:
        with st.sidebar:
            st.markdown(sidebar_description)


# ============================================================
# LOADING SKELETON CARDS
# ============================================================

def render_skeleton(n_rows: int = 3, n_cols: int = 1) -> None:
    """Render placeholder shimmering skeleton cards while data loads."""
    for _ in range(n_rows):
        cols = st.columns(n_cols) if n_cols > 1 else [st.container()]
        for c in cols:
            with c:
                st.markdown(
                    f"""
                    <div class="pro-card" style="opacity:.7;">
                        <div class="skeleton" style="height:14px;width:40%;margin-bottom:10px;"></div>
                        <div class="skeleton" style="height:28px;width:70%;margin-bottom:8px;"></div>
                        <div class="skeleton" style="height:12px;width:90%;margin-bottom:6px;"></div>
                        <div class="skeleton" style="height:12px;width:60%;"></div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

