"""
app.py
Egyptian Public Figures Search Engine — Streamlit Web Interface
Run with:  streamlit run app.py
"""
import os
import sys

# الحل البديل لـ _file_
try:
    current_dir = os.path.dirname(os.path.abspath(_file_))
except NameError:
    current_dir = os.getcwd()

sys.path.insert(0, current_dir)

import pandas as pd
import streamlit as st

# sys.path.insert(0, os.path.dirname(__file__))
from searc_engine import EgyptianFiguresSearchEngine

MODEL_DIR ='models'

# ──────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Egyptian Figures Search",
    page_icon="𓂀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────────
# CSS
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;600;700&family=Inter:wght@300;400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
h1, h2, h3 { font-family: 'Playfair Display', serif !important; }

.hero {
    background: linear-gradient(135deg, #12304f 0%, #1a5fa8 65%, #2476c9 100%);
    border-radius: 16px;
    padding: 28px 32px 22px;
    color: white;
    margin-bottom: 24px;
}
.hero h1  { color: white !important; margin: 0 0 6px; font-size: 1.85rem !important; }
.hero-sub { color: rgba(255,255,255,0.72); font-size: 14px; margin: 0; }

.rcard {
    border: 1px solid #e4e0d8;
    border-radius: 14px;
    padding: 22px 26px 18px;
    margin-bottom: 14px;
    background: #fafaf8;
    position: relative;
    transition: box-shadow .2s, transform .15s;
}
.rcard:hover { box-shadow: 0 6px 22px rgba(0,0,0,0.09); transform: translateY(-1px); }

.medal {
    position: absolute;
    top: -11px; left: 22px;
    width: 34px; height: 34px;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 15px; font-weight: 700;
    border: 2.5px solid white;
    box-shadow: 0 2px 7px rgba(0,0,0,0.18);
    z-index: 1;
}
.m1 { background: #FFD700; color: #7a5200; }
.m2 { background: #C0C0C0; color: #3a3a3a; }
.m3 { background: #CD7F32; color: #fff; }
.mx { background: #e6e3dc; color: #555; font-size: 12px; }

.bar-wrap { background: #eceae5; border-radius: 6px; height: 9px; overflow: hidden; margin: 10px 0 3px; }
.bar-fill  { height: 9px; border-radius: 6px; }

.pill {
    display: inline-block;
    font-size: 11px; font-weight: 600;
    padding: 2px 9px;
    border-radius: 10px;
    margin-right: 5px; margin-top: 5px;
}

.fbadge {
    display: inline-block;
    font-size: 12px; font-weight: 500;
    padding: 3px 11px;
    border-radius: 12px;
    margin-right: 4px;
}

.snip { font-size: 14px; color: #454545; line-height: 1.75; margin: 11px 0 9px; }
.rel-label { font-size: 11px; font-weight: 600; margin-top: 1px; }

.sbox { background:#f5f3ee; border-radius:10px; padding:13px; text-align:center; margin-bottom:10px; }
.sbox .num { font-size:22px; font-weight:700; color:#1a1a1a; }
.sbox .lbl { font-size:10px; color:#999; text-transform:uppercase; letter-spacing:.07em; }

.algo-box {
    border-left: 4px solid;
    border-radius: 6px;
    padding: 10px 14px;
    font-size: 13px;
    margin-bottom: 1rem;
    line-height: 1.5;
}
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────────────────────────────────────
FIELD_STYLE = {
    "Politics":   ("background:#dbeafe;color:#1e40af",  "🏛️"),
    "Literature": ("background:#ede9fe;color:#4c1d95",  "📚"),
    "Poetry":     ("background:#ede9fe;color:#5b21b6",  "✍️"),
    "Music":      ("background:#d1fae5;color:#065f46",  "🎵"),
    "Sports":     ("background:#fef3c7;color:#78350f",  "⚽"),
    "Science":    ("background:#dcfce7;color:#166534",  "🔬"),
    "Diplomacy":  ("background:#fde8d8;color:#7c2d12",  "🌍"),
    "Cinema":     ("background:#fce7f3;color:#701a75",  "🎬"),
    "Technology": ("background:#e0f2fe;color:#0c4a6e",  "💻"),
    "Religion":   ("background:#fef9c3;color:#713f12",  "☪️"),
    "History":    ("background:#f3f4f6;color:#374151",  "🏺"),
    "Military":   ("background:#fee2e2;color:#7f1d1d",  "⚔️"),
}

ALGO_META = {
    "BM25": {
        "color": "#1a5fa8",
        "desc": "**BM25** — Probabilistic model. Applies TF saturation and document-length normalization. Best general-purpose ranker for short queries.",
        "bg": "#dbeafe",
    },
    "TF-IDF": {
        "color": "#7c3aed",
        "desc": "**TF-IDF** — Vector-space cosine similarity. Weights rare terms more heavily. Works well for specific, uncommon keywords.",
        "bg": "#ede9fe",
    },
    "Hybrid": {
        "color": "#065f46",
        "desc": "**Hybrid (60% BM25 + 40% TF-IDF)** — Min-max normalized combination. Balances both models for robust, stable ranking.",
        "bg": "#d1fae5",
    },
}

SUGGESTIONS = [
    ("Nobel Prize", "🔬"),
    ("football Africa Cup", "⚽"),
    ("Egyptian actress cinema", "🎬"),
    ("Arab singer music", "🎵"),
    ("revolution president", "🏛️"),
    ("squash world champion", "🏆"),
    ("ancient pharaoh Egypt", "🏺"),
    ("feminist writer activist", "📚"),
]


# ──────────────────────────────────────────────────────────────────────────────
# LOAD MODEL
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="⏳ Loading search engine…")
def load_engine():
    return EgyptianFiguresSearchEngine.load(MODEL_DIR)

engine = load_engine()
stats  = engine.stats()


# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────
def medal_html(rank: int) -> str:
    cls  = {1: "m1", 2: "m2", 3: "m3"}.get(rank, "mx")
    icon = {1: "🥇", 2: "🥈", 3: "🥉"}.get(rank, str(rank))
    return f'<div class="medal {cls}">{icon}</div>'

def bar_html(pct: int, color: str, height: int = 9) -> str:
    return (f'<div class="bar-wrap" style="height:{height}px">'
            f'<div class="bar-fill" style="width:{pct}%;background:{color};height:{height}px"></div>'
            f'</div>')

def pill_html(label: str, bg: str, color: str) -> str:
    return f'<span class="pill" style="background:{bg};color:{color}">{label}</span>'

def rel_info(pct: int):
    if pct >= 80:  return "Highly Relevant",     "#065f46"
    if pct >= 55:  return "Relevant",             "#1a5fa8"
    if pct >= 30:  return "Somewhat Relevant",    "#92400e"
    return              "Marginally Relevant",    "#6b7280"


# ──────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 𓂀 Egyptian Search")
    st.markdown("---")

    st.markdown("### 🧮 Ranking Algorithm")
    algo = st.radio("", list(ALGO_META.keys()), index=0)
    m     = ALGO_META[algo]
    color = m["color"]

    st.markdown(
        f'<div class="algo-box" style="border-color:{color};background:{color}14;color:{color}">'
        f'{m["desc"]}</div>',
        unsafe_allow_html=True,
    )

    if algo == "Hybrid":
        alpha = st.slider("BM25 weight (α)", 0.1, 0.9, 0.6, 0.05)
    else:
        alpha = 0.6

    st.markdown("---")
    st.markdown("### 🎯 Filters")
    sel_fields = st.multiselect("Field", sorted(stats["fields"]), default=[])
    sel_eras   = st.multiselect("Era",   sorted(stats["eras"]),   default=[])

    st.markdown("---")
    st.markdown("### 📊 Corpus")
    c1, c2 = st.columns(2)
    for col, num, lbl in [
        (c1, stats["num_documents"], "Figures"),
        (c2, stats["vocab_size"],    "Vocab"),
        (c1, len(stats["fields"]),   "Fields"),
        (c2, stats["avg_doc_length"],"Avg Len"),
    ]:
        col.markdown(
            f'<div class="sbox"><div class="num">{num}</div><div class="lbl">{lbl}</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown("### 💡 Try these")
    btn_cols = st.columns(2)
    for j, (s, icon) in enumerate(SUGGESTIONS):
        if btn_cols[j % 2].button(f"{icon} {s}", use_container_width=True, key=f"sg_{j}"):
            st.session_state["q"] = s
            st.rerun()

    st.markdown("---")
    st.markdown(
        '<div style="font-size:11px;color:#aaa;text-align:center">'
        'Built with TF-IDF & BM25 · Wikipedia data</div>',
        unsafe_allow_html=True,
    )


# ──────────────────────────────────────────────────────────────────────────────
# MAIN AREA
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>𓂀 Egyptian Public Figures Search Engine</h1>
    <p class="hero-sub">100 Egyptian public figures · Ranked by TF-IDF & BM25 · Data from Wikipedia</p>
</div>
""", unsafe_allow_html=True)

query = st.text_input(
    label="",
    value=st.session_state.get("q", ""),
    placeholder="e.g.  Nobel Prize,  actress cinema,  football player,  ancient pharaoh …",
    key="q",
)

row1, row2, row3 = st.columns([1, 2, 2])
with row1:
    search_btn = st.button("🔍 Search", type="primary", use_container_width=True)
with row2:
    top_k = st.slider("Max results", 3, 20, 8, 1)
with row3:
    show_cmp = st.toggle("📊 Score comparison table", value=False)


# ──────────────────────────────────────────────────────────────────────────────
# SEARCH & DISPLAY
# ──────────────────────────────────────────────────────────────────────────────
if query and query.strip():
    method_map = {"TF-IDF": "tfidf", "BM25": "bm25", "Hybrid": "hybrid"}
    method = method_map[algo]

    with st.spinner("Ranking…"):
        all_results = engine.search(query, method=method, top_k=100, alpha=alpha)

    if sel_fields:
        all_results = [r for r in all_results if r["field"] in sel_fields]
    if sel_eras:
        all_results = [r for r in all_results if r["era"] in sel_eras]

    results   = all_results[:top_k]
    n         = len(results)
    max_score = results[0]["score"] if results else 1.0

    st.markdown("---")

    if not results:
        st.error("No results found. Try different keywords or remove filters.")
        st.stop()

    filt_str = ""
    parts = []
    if sel_fields: parts.append(", ".join(sel_fields))
    if sel_eras:   parts.append(", ".join(sel_eras))
    if parts:      filt_str = f" · filtered by **{' / '.join(parts)}**"

    st.markdown(
        f"**{n} result{'s' if n != 1 else ''}** for *\"{query}\"* "
        f"— ranked by <span style='color:{color};font-weight:700'>{algo}</span>{filt_str}",
        unsafe_allow_html=True,
    )

    # ── Score comparison table ────────────────────────────────────────────────
    if show_cmp and results:
        max_bm25  = max(r["bm25_score"]  for r in results) or 1
        max_tfidf = max(r["tfidf_score"] for r in results) or 1
        max_fin   = max(r["score"]       for r in results) or 1

        table_data = []
        for i, r in enumerate(results, 1):
            table_data.append({
                "Rank":        f"#{i}",
                "Name":        r["name"],
                "Field":       r["field"],
                "BM25":        r["bm25_score"],
                "TF-IDF":      r["tfidf_score"],
                "Final Score": r["score"],
            })

        df = pd.DataFrame(table_data)
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "BM25":        st.column_config.ProgressColumn("BM25",        min_value=0, max_value=max_bm25,  format="%.4f"),
                "TF-IDF":      st.column_config.ProgressColumn("TF-IDF",      min_value=0, max_value=max_tfidf, format="%.4f"),
                "Final Score": st.column_config.ProgressColumn("Final Score", min_value=0, max_value=max_fin,   format="%.4f"),
            }
        )
        st.markdown("<br>", unsafe_allow_html=True)

    # ── Result cards ─────────────────────────────────────────────────────────
    max_bm25_all  = max(r["bm25_score"]  for r in results) or 1
    max_tfidf_all = max(r["tfidf_score"] for r in results) or 1

    for rank, r in enumerate(results, 1):
        field_s, field_i = FIELD_STYLE.get(r["field"], ("background:#eee;color:#333", "👤"))
        snippet  = engine.get_snippet(r["extract"], query, window=290)
        f_pct    = int(r["score"] / max_score * 100)
        b_pct    = int(r["bm25_score"]  / max_bm25_all  * 100)
        t_pct    = int(r["tfidf_score"] / max_tfidf_all * 100)
        rel_lbl, rel_col = rel_info(f_pct)

        pills = (
            pill_html(f"BM25 {r['bm25_score']:.3f}",    "#dbeafe", "#1a5fa8") +
            pill_html(f"TF-IDF {r['tfidf_score']:.3f}", "#ede9fe", "#7c3aed") +
            pill_html(f"{algo} {r['score']:.4f}",        m["bg"],   color)
        )

        st.markdown(f"""
        <div class="rcard">
          {medal_html(rank)}
          <div style="display:flex;justify-content:space-between;align-items:flex-start;
                      flex-wrap:wrap;gap:8px;padding-top:10px">
            <div>
              <span style="font-family:'Playfair Display',serif;font-size:19px;
                           font-weight:700;color:#111">{r['name']}</span>
              &nbsp;
              <span class="fbadge" style="{field_s}">{field_i} {r['field']}</span>
              <span class="fbadge" style="background:#f0ede5;color:#666">{r['era']}</span>
            </div>
            <div style="text-align:right">
              <div style="font-size:14px;font-weight:700;color:{color}">{algo}: {r['score']:.4f}</div>
              <div class="rel-label" style="color:{rel_col}">{rel_lbl}</div>
            </div>
          </div>

          <div style="display:flex;justify-content:space-between;
                      font-size:11px;color:#aaa;margin-top:10px;margin-bottom:2px">
            <span style="font-weight:600;color:#888">Rank #{rank} of {n}</span>
            <span style="color:{color};font-weight:700">{f_pct}% of top score</span>
          </div>
          {bar_html(f_pct, color)}
          <div style="display:flex;justify-content:space-between;font-size:10px;color:#ccc;margin-bottom:6px">
            <span>0</span><span>50</span><span>100%</span>
          </div>

          <div style="display:flex;align-items:center;gap:8px;margin-bottom:3px">
            <span style="font-size:11px;color:#1a5fa8;width:52px;font-weight:600">BM25</span>
            {bar_html(b_pct, '#1a5fa8', 6)}
            <span style="font-size:11px;color:#1a5fa8;min-width:40px">{r['bm25_score']:.3f}</span>
          </div>
          <div style="display:flex;align-items:center;gap:8px;margin-bottom:10px">
            <span style="font-size:11px;color:#7c3aed;width:52px;font-weight:600">TF-IDF</span>
            {bar_html(t_pct, '#7c3aed', 6)}
            <span style="font-size:11px;color:#7c3aed;min-width:40px">{r['tfidf_score']:.3f}</span>
          </div>

          <div>{pills}</div>
          <div class="snip">{snippet}</div>
          <a href="{r['wikipedia_url']}" target="_blank"
             style="color:#1a5fa8;text-decoration:none;font-size:13px;font-weight:500">
            → Read on Wikipedia
          </a>
        </div>
        """, unsafe_allow_html=True)

else:
    st.markdown("""
    <div style="text-align:center;padding:70px 20px;color:#999">
      <div style="font-size:52px;margin-bottom:18px">𓂀</div>
      <div style="font-size:19px;color:#555;font-weight:600;margin-bottom:8px">
        Search 100 Egyptian public figures
      </div>
      <div style="font-size:14px;line-height:1.8">
        Ranked by <span style="color:#1a5fa8;font-weight:600">BM25</span>,
        <span style="color:#7c3aed;font-weight:600">TF-IDF</span>, or
        <span style="color:#065f46;font-weight:600">Hybrid</span> scoring
        &nbsp;·&nbsp; 12 fields &nbsp;·&nbsp; Wikipedia data
      </div>
    </div>
    """, unsafe_allow_html=True)
