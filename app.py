import streamlit as st
import joblib
import numpy as np
import re
import time
import html as html_module

# ──────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SENTRY · Message Threat Scanner",
    page_icon="◈",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ──────────────────────────────────────────────────────────────────────────
# MODEL LOADING
# ──────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_pipeline():
    model = joblib.load("model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
    return model, vectorizer

model, vectorizer = load_pipeline()
VOCAB_INV = {idx: term for term, idx in vectorizer.vocabulary_.items()}
COEF = model.coef_[0]  # positive -> pushes toward spam (class 1), negative -> ham (class 0)

# class 0 = ham, class 1 = spam (confirmed directly from spam_ham_dataset.csv's
# own label_num column: label_num=0 -> 'ham', label_num=1 -> 'spam', and verified
# against the trained model's predictions on known examples)
SPAM_LABEL, HAM_LABEL = 1, 0

# ──────────────────────────────────────────────────────────────────────────
# STYLES — "SENTRY" signal-scanner identity
# Palette: void black base, signal-green clear / alarm-red threat,
# slate panels, monospace readouts paired with a clean grotesk body.
# ──────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700;800&family=Inter:wght@400;500;600;700&display=swap');

:root {
    --void: #090B10;
    --panel: #11141C;
    --panel-raised: #161A24;
    --line: #232838;
    --line-bright: #313850;
    --text: #E7EAF2;
    --text-dim: #8A93A8;
    --text-faint: #565F75;
    --signal: #39FF8E;
    --signal-dim: #1C8C52;
    --alarm: #FF3B5C;
    --alarm-dim: #8C1F33;
    --amber: #FFB454;
}

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp {
    background: radial-gradient(ellipse 80% 50% at 50% -10%, #131826 0%, var(--void) 60%);
    color: var(--text);
}

#MainMenu, header, footer { visibility: hidden; }
.block-container { padding-top: 2.2rem; max-width: 760px; }

/* ── Scanline ambient overlay ── */
.stApp::before {
    content: "";
    position: fixed;
    top: 0; left: 0; width: 100%; height: 100%;
    background: repeating-linear-gradient(
        to bottom,
        rgba(57,255,142,0.012) 0px,
        rgba(57,255,142,0.012) 1px,
        transparent 1px,
        transparent 3px
    );
    pointer-events: none;
    z-index: 0;
}

/* ── Header / masthead ── */
.sentry-mast {
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    border-bottom: 1px solid var(--line);
    padding-bottom: 14px;
    margin-bottom: 6px;
}
.sentry-logo {
    font-family: 'JetBrains Mono', monospace;
    font-weight: 800;
    font-size: 1.5rem;
    letter-spacing: 0.08em;
    color: var(--text);
}
.sentry-logo span { color: var(--signal); }
.sentry-status {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    color: var(--text-faint);
    letter-spacing: 0.06em;
    text-transform: uppercase;
}
.sentry-status .dot {
    display: inline-block;
    width: 6px; height: 6px;
    border-radius: 50%;
    background: var(--signal);
    margin-right: 6px;
    box-shadow: 0 0 8px var(--signal);
    animation: pulse 2s ease-in-out infinite;
}
@keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.35; } }

.sentry-sub {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    color: var(--text-faint);
    margin: 10px 0 28px 0;
    letter-spacing: 0.03em;
}

/* ── Input panel ── */
.stTextArea textarea {
    background: var(--panel) !important;
    border: 1px solid var(--line) !important;
    border-radius: 6px !important;
    color: var(--text) !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.92rem !important;
    line-height: 1.6 !important;
    padding: 16px !important;
    transition: border-color 0.2s ease;
}
.stTextArea textarea:focus {
    border-color: var(--signal-dim) !important;
    box-shadow: 0 0 0 1px var(--signal-dim), 0 0 24px rgba(57,255,142,0.06) !important;
}
.stTextArea label { display: none; }
.stTextArea textarea::placeholder { color: var(--text-faint) !important; }

/* ── Scan button ── */
div.stButton > button {
    width: 100%;
    background: var(--signal) !important;
    color: #06120A !important;
    border: none !important;
    border-radius: 6px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-weight: 700 !important;
    font-size: 0.85rem !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    padding: 0.7rem 0 !important;
    margin-top: 10px;
    transition: all 0.15s ease !important;
    box-shadow: 0 0 0 rgba(57,255,142,0);
}
div.stButton > button * {
    color: #06120A !important;
}
div.stButton > button:hover {
    background: #4DFFA0 !important;
    box-shadow: 0 0 28px rgba(57,255,142,0.35) !important;
    transform: translateY(-1px);
}
div.stButton > button:active { transform: translateY(0px); }

/* ── Sample chips row ── */
.sample-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    color: var(--text-faint);
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin: 18px 0 8px 2px;
}
div[data-testid="column"] div.stButton > button {
    background: var(--panel) !important;
    color: var(--text-dim) !important;
    border: 1px solid var(--line) !important;
    font-weight: 500 !important;
    font-size: 0.74rem !important;
    letter-spacing: 0.02em !important;
    text-transform: none !important;
    padding: 0.5rem 0.6rem !important;
    margin-top: 0;
}
div[data-testid="column"] div.stButton > button * {
    color: var(--text-dim) !important;
}
div[data-testid="column"] div.stButton > button:hover {
    border-color: var(--line-bright) !important;
    color: var(--text) !important;
    background: var(--panel-raised) !important;
    box-shadow: none !important;
    transform: none;
}
div[data-testid="column"] div.stButton > button:hover * {
    color: var(--text) !important;
}

/* ── Verdict card ── */
.verdict-card {
    margin-top: 28px;
    border-radius: 10px;
    padding: 28px 28px 24px 28px;
    position: relative;
    overflow: hidden;
    animation: rise 0.45s cubic-bezier(0.16,1,0.3,1);
}
@keyframes rise { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }

.verdict-spam { background: linear-gradient(135deg, rgba(255,59,92,0.10), rgba(17,20,28,0.4)); border: 1px solid rgba(255,59,92,0.35); }
.verdict-ham  { background: linear-gradient(135deg, rgba(57,255,142,0.08), rgba(17,20,28,0.4)); border: 1px solid rgba(57,255,142,0.30); }

.verdict-eyebrow {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    margin-bottom: 6px;
}
.verdict-spam .verdict-eyebrow { color: var(--alarm); }
.verdict-ham .verdict-eyebrow { color: var(--signal); }

.verdict-title {
    font-family: 'JetBrains Mono', monospace;
    font-weight: 800;
    font-size: 2.1rem;
    letter-spacing: 0.01em;
    line-height: 1.1;
    margin-bottom: 4px;
}
.verdict-spam .verdict-title { color: var(--alarm); }
.verdict-ham .verdict-title { color: var(--signal); }

.verdict-desc {
    font-size: 0.88rem;
    color: var(--text-dim);
    margin-bottom: 20px;
    max-width: 480px;
}

/* confidence meter */
.meter-row { display: flex; align-items: center; gap: 14px; margin-bottom: 4px; }
.meter-track {
    flex: 1; height: 8px; border-radius: 4px;
    background: rgba(255,255,255,0.06);
    overflow: hidden;
    position: relative;
}
.meter-fill { height: 100%; border-radius: 4px; transition: width 0.6s cubic-bezier(0.16,1,0.3,1); }
.verdict-spam .meter-fill { background: linear-gradient(90deg, var(--alarm-dim), var(--alarm)); }
.verdict-ham .meter-fill { background: linear-gradient(90deg, var(--signal-dim), var(--signal)); }
.meter-pct {
    font-family: 'JetBrains Mono', monospace;
    font-weight: 700;
    font-size: 1rem;
    min-width: 58px;
    text-align: right;
}
.verdict-spam .meter-pct { color: var(--alarm); }
.verdict-ham .meter-pct { color: var(--signal); }
.meter-caption {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    color: var(--text-faint);
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-top: 6px;
}

/* ── Signal breakdown ── */
.signal-section { margin-top: 26px; }
.signal-heading {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    color: var(--text-faint);
    letter-spacing: 0.14em;
    text-transform: uppercase;
    margin-bottom: 12px;
    display: flex;
    align-items: center;
    gap: 8px;
}
.signal-heading::after {
    content: "";
    flex: 1;
    height: 1px;
    background: var(--line);
}

.token-row {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 7px 0;
    border-bottom: 1px solid rgba(255,255,255,0.04);
}
.token-word {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.84rem;
    font-weight: 600;
    min-width: 110px;
    color: var(--text);
}
.token-bar-track {
    flex: 1;
    height: 5px;
    border-radius: 3px;
    background: rgba(255,255,255,0.05);
    position: relative;
    overflow: hidden;
}
.token-bar-fill { position: absolute; top:0; bottom:0; border-radius: 3px; }
.token-bar-fill.spamward { background: var(--alarm); right: 50%; }
.token-bar-fill.hamward { background: var(--signal); left: 50%; }
.token-weight {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    min-width: 52px;
    text-align: right;
    color: var(--text-faint);
}

/* highlighted message readback */
.readback {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.82rem;
    line-height: 1.7;
    color: var(--text-dim);
    background: var(--panel);
    border: 1px solid var(--line);
    border-radius: 6px;
    padding: 16px;
    margin-top: 10px;
}
.hl-spam { background: rgba(255,59,92,0.18); color: #FF8FA3; border-radius: 3px; padding: 1px 3px; font-weight: 600; }
.hl-ham { background: rgba(57,255,142,0.16); color: #8FFFC1; border-radius: 3px; padding: 1px 3px; font-weight: 600; }

/* metrics strip */
.metrics-strip {
    display: flex;
    gap: 1px;
    margin-top: 28px;
    border: 1px solid var(--line);
    border-radius: 8px;
    overflow: hidden;
}
.metric-cell {
    flex: 1;
    background: var(--panel);
    padding: 14px 16px;
    text-align: center;
}
.metric-cell + .metric-cell { border-left: 1px solid var(--line); }
.metric-val {
    font-family: 'JetBrains Mono', monospace;
    font-weight: 700;
    font-size: 1.15rem;
    color: var(--text);
}
.metric-lbl {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.62rem;
    color: var(--text-faint);
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-top: 3px;
}

/* footer */
.sentry-footer {
    margin-top: 50px;
    padding-top: 16px;
    border-top: 1px solid var(--line);
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    color: var(--text-faint);
    letter-spacing: 0.04em;
    display: flex;
    justify-content: space-between;
}

/* hide streamlit's default empty-state spacing oddities */
div[data-testid="stVerticalBlock"] > div:has(> div.element-container > div.stMarkdown) { }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────
# HEADER
# ──────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="sentry-mast">
    <div class="sentry-logo">◈ SENTRY<span>/</span>SCAN</div>
    <div class="sentry-status"><span class="dot"></span>MODEL ONLINE</div>
</div>
<div class="sentry-sub">MESSAGE THREAT SCANNER · TF-IDF + LOGISTIC REGRESSION · 45,256-TERM VOCABULARY</div>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────
# SAMPLE MESSAGES
# ──────────────────────────────────────────────────────────────────────────
SAMPLES = {
    "Phishing email": "URGENT: Your bank account has been temporarily suspended due to suspicious activity. Verify your account immediately at https://secure-bank-login.com to avoid permanent closure.",
    "Office email": "Hi team, attached is the updated project schedule for next week. Let me know if the Tuesday meeting time still works for everyone.",
    "Promo blast": "CONGRATULATIONS! You have been selected for a FREE prize offer. Click here now to claim your cash reward before it expires!",
    "Quick note": "Thanks for sending that over, I will review the attached document this morning and get back to you with comments.",
}

if "message_input" not in st.session_state:
    st.session_state.message_input = ""

st.markdown('<div class="sample-label">LOAD SAMPLE TRANSMISSION</div>', unsafe_allow_html=True)
cols = st.columns(len(SAMPLES))
for col, (label, text) in zip(cols, SAMPLES.items()):
    with col:
        if st.button(label, key=f"sample_{label}", use_container_width=True):
            st.session_state.message_input = text
            st.rerun()

# ──────────────────────────────────────────────────────────────────────────
# INPUT
# ──────────────────────────────────────────────────────────────────────────
message = st.text_area(
    "message",
    height=130,
    placeholder="Paste or type the message text to scan for spam signals...",
    key="message_input",
    label_visibility="collapsed",
)

scan_clicked = st.button("▶  RUN SCAN", use_container_width=True)

# ──────────────────────────────────────────────────────────────────────────
# ANALYSIS
# ──────────────────────────────────────────────────────────────────────────
def get_token_contributions(text, top_n=8):
    """Tokenize using the vectorizer's own pattern and rank vocab tokens
    present in the message by their model coefficient (signed contribution)."""
    tokenizer_pattern = re.compile(vectorizer.token_pattern if hasattr(vectorizer, "token_pattern") else r"(?u)\b\w\w+\b")
    raw_tokens = tokenizer_pattern.findall(text.lower())
    seen = {}
    for tok in raw_tokens:
        if tok in vectorizer.vocabulary_:
            idx = vectorizer.vocabulary_[tok]
            seen[tok] = COEF[idx]  # positive = spamward, negative = hamward
    ranked = sorted(seen.items(), key=lambda kv: abs(kv[1]), reverse=True)
    return ranked[:top_n], raw_tokens

def render_highlighted(text, contributions):
    """Wrap matched tokens in the original text with highlight spans."""
    contrib_map = {w: weight for w, weight in contributions}
    tokenizer_pattern = re.compile(vectorizer.token_pattern if hasattr(vectorizer, "token_pattern") else r"(?u)\b\w\w+\b")

    out = []
    last_end = 0
    for m in tokenizer_pattern.finditer(text):
        word_lower = m.group(0).lower()
        out.append(html_module.escape(text[last_end:m.start()]))
        seg = html_module.escape(text[m.start():m.end()])
        if word_lower in contrib_map:
            cls = "hl-spam" if contrib_map[word_lower] > 0 else "hl-ham"
            out.append(f'<span class="{cls}">{seg}</span>')
        else:
            out.append(seg)
        last_end = m.end()
    out.append(html_module.escape(text[last_end:]))
    return "".join(out)

if scan_clicked:
    if not message.strip():
        st.warning("Enter a message to scan — there's nothing in the queue.")
    else:
        progress_ph = st.empty()
        steps = ["TOKENIZING INPUT…", "VECTORIZING AGAINST 45,256-TERM INDEX…", "SCORING WITH CLASSIFIER…"]
        for step in steps:
            progress_ph.markdown(
                f'<div style="font-family:\'JetBrains Mono\',monospace; font-size:0.72rem; color:#39FF8E; letter-spacing:0.08em; margin-top:14px;">▸ {step}</div>',
                unsafe_allow_html=True,
            )
            time.sleep(0.22)
        progress_ph.empty()

        X = vectorizer.transform([message])
        pred = model.predict(X)[0]
        proba = model.predict_proba(X)[0]
        spam_prob = proba[SPAM_LABEL]
        ham_prob = proba[HAM_LABEL]
        is_spam = pred == SPAM_LABEL

        confidence = spam_prob if is_spam else ham_prob
        confidence_pct = round(confidence * 100, 1)

        tokenizer_pat = vectorizer.token_pattern if hasattr(vectorizer, "token_pattern") else r"(?u)\b\w\w+\b"
        all_message_tokens = re.findall(tokenizer_pat, message.lower())
        n_tokens_matched = sum(1 for t in all_message_tokens if t in vectorizer.vocabulary_)
        total_tokens = len(all_message_tokens)

        contributions, all_tokens = get_token_contributions(message, top_n=8)

        # ── Verdict card ──
        if is_spam:
            verdict_class = "verdict-spam"
            eyebrow = "⚠ THREAT DETECTED"
            title = "SPAM"
            desc = "This message carries strong markers consistent with unsolicited or scam content."
        else:
            verdict_class = "verdict-ham"
            eyebrow = "✓ NO THREAT FOUND"
            title = "CLEAR"
            desc = "This message reads as ordinary personal correspondence — no spam signal detected."

        st.markdown(f"""
        <div class="verdict-card {verdict_class}">
            <div class="verdict-eyebrow">{eyebrow}</div>
            <div class="verdict-title">{title}</div>
            <div class="verdict-desc">{desc}</div>
            <div class="meter-row">
                <div class="meter-track"><div class="meter-fill" style="width:{confidence_pct}%;"></div></div>
                <div class="meter-pct">{confidence_pct}%</div>
            </div>
            <div class="meter-caption">CONFIDENCE · SPAM {round(spam_prob*100,1)}% / HAM {round(ham_prob*100,1)}%</div>
        </div>
        """, unsafe_allow_html=True)

        # ── Metrics strip ──
        st.markdown(f"""
        <div class="metrics-strip">
            <div class="metric-cell"><div class="metric-val">{total_tokens}</div><div class="metric-lbl">Tokens Read</div></div>
            <div class="metric-cell"><div class="metric-val">{n_tokens_matched}</div><div class="metric-lbl">In Vocabulary</div></div>
            <div class="metric-cell"><div class="metric-val">{len(contributions)}</div><div class="metric-lbl">Signal Words</div></div>
        </div>
        """, unsafe_allow_html=True)

        # ── Token contribution breakdown ──
        if contributions:
            st.markdown('<div class="signal-section"><div class="signal-heading">SIGNAL BREAKDOWN — STRONGEST CONTRIBUTING WORDS</div></div>', unsafe_allow_html=True)
            max_abs = max(abs(w) for _, w in contributions) or 1.0
            rows_html = ""
            for word, weight in contributions:
                pct = min(abs(weight) / max_abs * 50, 50)
                bar_cls = "spamward" if weight > 0 else "hamward"
                bar_style = f"width:{pct}%;"
                direction = "→ spam" if weight > 0 else "→ ham"
                color = "#FF3B5C" if weight > 0 else "#39FF8E"
                rows_html += f"""
                <div class="token-row">
                    <div class="token-word">{html_module.escape(word)}</div>
                    <div class="token-bar-track"><div class="token-bar-fill {bar_cls}" style="{bar_style}"></div></div>
                    <div class="token-weight" style="color:{color};">{direction}</div>
                </div>
                """
            st.markdown(rows_html, unsafe_allow_html=True)

            # ── Highlighted readback ──
            st.markdown('<div class="signal-section"><div class="signal-heading">ANNOTATED TRANSMISSION</div></div>', unsafe_allow_html=True)
            highlighted = render_highlighted(message, contributions)
            st.markdown(f'<div class="readback">{highlighted}</div>', unsafe_allow_html=True)
        else:
            st.markdown(
                '<div style="font-family:\'JetBrains Mono\',monospace; font-size:0.78rem; color:#565F75; margin-top:20px;">'
                'No individual words in this message matched strong signal terms in the vocabulary — the verdict reflects the overall TF-IDF pattern.'
                '</div>',
                unsafe_allow_html=True,
            )

# ──────────────────────────────────────────────────────────────────────────
# FOOTER
# ──────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="sentry-footer">
    <span>SENTRY/SCAN v1.0</span>
    <span>LOGISTIC REGRESSION · scikit-learn</span>
</div>
""", unsafe_allow_html=True)