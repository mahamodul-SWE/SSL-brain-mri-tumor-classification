APP_CSS = r"""
<style>
html, body, [class*="css"] { font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; }

.stApp {
    background:
      radial-gradient(circle at 10% 3%, rgba(37, 99, 235, .10), transparent 24rem),
      radial-gradient(circle at 90% 12%, rgba(14, 165, 233, .08), transparent 25rem),
      #f7f9fc;
}

.block-container {
    max-width: 1180px;
    padding-top: 1.2rem;
    padding-bottom: 3rem;
}

.hero {
    position: relative;
    overflow: hidden;
    border-radius: 26px;
    padding: 2rem 2.25rem;
    margin-bottom: 1.25rem;
    color: white;
    background: linear-gradient(125deg, #0f172a 0%, #172554 48%, #0c4a6e 100%);
    box-shadow: 0 22px 55px rgba(15,23,42,.17);
}
.hero::before {
    content: '';
    position: absolute;
    width: 360px;
    height: 360px;
    right: -135px;
    top: -170px;
    border-radius: 50%;
    background: rgba(56,189,248,.16);
}
.hero-kicker {
    position: relative;
    z-index: 1;
    font-size: .76rem;
    font-weight: 800;
    letter-spacing: .15em;
    text-transform: uppercase;
    color: #7dd3fc;
    margin-bottom: .6rem;
}
.hero-title-row {
    position: relative;
    z-index: 1;
    display: flex;
    justify-content: space-between;
    gap: 2rem;
    align-items: center;
}
.hero h1 {
    margin: 0;
    font-size: clamp(2rem, 4vw, 3.3rem);
    line-height: 1.04;
    letter-spacing: -.045em;
}
.hero p {
    max-width: 760px;
    margin: .85rem 0 0;
    color: #dbeafe;
    font-size: .98rem;
    line-height: 1.7;
}
.hero-mark {
    font-size: 4.6rem;
    opacity: .9;
    filter: drop-shadow(0 10px 25px rgba(0,0,0,.2));
}
.hero-badges {
    position: relative;
    z-index: 1;
    margin-top: 1.05rem;
    display: flex;
    gap: .5rem;
    flex-wrap: wrap;
}
.hero-badge {
    border: 1px solid rgba(255,255,255,.18);
    background: rgba(255,255,255,.08);
    padding: .4rem .7rem;
    border-radius: 999px;
    font-size: .76rem;
    font-weight: 650;
}

.result-card {
    border-radius: 20px;
    padding: 1.3rem 1.4rem;
    border: 1px solid #dbe4f0;
    background: white;
    box-shadow: 0 12px 35px rgba(15,23,42,.065);
    min-height: 150px;
}
.result-label {
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: .09em;
    font-size: .7rem;
    font-weight: 800;
}
.result-value {
    color: #0f172a;
    font-size: 1.58rem;
    line-height: 1.2;
    font-weight: 800;
    margin-top: .35rem;
}
.result-sub { color: #64748b; margin-top: .45rem; font-size: .84rem; }
.positive { border-top: 5px solid #e11d48; }
.negative { border-top: 5px solid #16a34a; }
.neutral { border-top: 5px solid #2563eb; }
.review { border-top: 5px solid #f59e0b; }

.status-pill {
    display: inline-flex;
    align-items: center;
    gap: .4rem;
    padding: .38rem .65rem;
    border-radius: 999px;
    font-size: .77rem;
    font-weight: 750;
    margin-top: .65rem;
}
.status-good { background: #dcfce7; color: #166534; }
.status-warn { background: #fef3c7; color: #92400e; }
.status-bad { background: #ffe4e6; color: #9f1239; }

.control-panel {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 14px;
    padding: .85rem .95rem;
    color: #475569;
    font-size: .83rem;
    line-height: 1.55;
    margin-bottom: .8rem;
}

.selection-strip {
    display: inline-block;
    margin: .2rem 0 .75rem;
    padding: .45rem .75rem;
    border-radius: 999px;
    background: #eff6ff;
    border: 1px solid #bfdbfe;
    color: #1d4ed8;
    font-size: .82rem;
}

.upload-help {
    text-align: center;
    padding: .45rem 0 .2rem;
    color: #64748b;
    font-size: .86rem;
}

.small-note {
    color: #64748b;
    font-size: .8rem;
    line-height: 1.6;
}

.footer-note {
    margin-top: 2rem;
    padding: 1rem 1.2rem;
    border-radius: 16px;
    background: #fff7ed;
    border: 1px solid #fed7aa;
    color: #9a3412;
    font-size: .83rem;
    line-height: 1.65;
}

[data-testid="stSidebar"] { background: #0f172a; }
[data-testid="stSidebar"] * { color: #2afafb; }
[data-testid="stSidebar"] [data-testid="stMetric"] {
    background: rgba(255,255,255,.06);
    border-color: rgba(255,255,255,.1);
}
[data-testid="stSidebar"] .stButton button { width: 100%; }

.stButton > button {
    border-radius: 12px;
    border: none;
    font-weight: 700;
    padding: .62rem 1rem;
    background: linear-gradient(120deg, #2563eb, #0284c7);
    color: white;
    box-shadow: 0 8px 18px rgba(37,99,235,.18);
    transition: transform .15s ease, filter .15s ease;
}
.stButton > button:hover {
    border: none;
    color: white;
    filter: brightness(1.05);
    transform: translateY(-1px);
}

[data-testid="stFileUploader"] {
    background: white;
    border-radius: 18px;
    padding: .55rem;
    border: 1px dashed #94a3b8;
}

[data-testid="stMetric"] {
    background: white;
    border: 1px solid #e5eaf2;
    padding: .85rem 1rem;
    border-radius: 16px;
    box-shadow: 0 8px 25px rgba(15,23,42,.05);
}

.stDataFrame {
    border-radius: 14px;
    overflow: hidden;
}

.stTabs [data-baseweb="tab-list"] { gap: .35rem; }
.stTabs [data-baseweb="tab"] {
    border-radius: 10px;
    padding: .55rem .85rem;
    font-weight: 650;
}

@media (max-width: 720px) {
    .hero { padding: 1.45rem 1.15rem; border-radius: 18px; }
    .hero-title-row { display: block; }
    .hero-mark { display: none; }
    .block-container { padding-left: .85rem; padding-right: .85rem; }
}



[data-testid="stSidebar"] code {
    background: #2563eb !important;
    color: #2afafb !important;
    -webkit-text-fill-color: #2afafb !important;
    border-radius: 6px !important;
    padding: 0.1rem 0.3rem !important;
}

[data-testid="stHeader"] {
    background-color: #BDF8FA !important;
}


</style>
"""
