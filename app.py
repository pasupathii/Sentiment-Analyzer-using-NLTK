
import os
import json
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import joblib

from data_loader import clean_text, LABEL_NAMES, LABEL_MAP, ensure_nltk_data
from explainer import explain_prediction

# ---------------------------------------------------------------------------
# Page config & constants
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Sentiment Analyzer Pro",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

SENTIMENT_CONFIG = {
    "positive": {"emoji": "😊", "color": "#22C55E", "gradient": "linear-gradient(135deg, #22C55E, #16A34A)"},
    "negative": {"emoji": "😞", "color": "#EF4444", "gradient": "linear-gradient(135deg, #EF4444, #DC2626)"},
    "neutral":  {"emoji": "😐", "color": "#F59E0B", "gradient": "linear-gradient(135deg, #F59E0B, #D97706)"},
}

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------

st.markdown("""
<style>
    /* ── Import font ─────────────────────────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* ── Global ──────────────────────────────────────────────────── */
    * { font-family: 'Inter', sans-serif !important; }

    .stApp {
        background: linear-gradient(180deg, #F8FAFC 0%, #E0F2FE 100%);
    }

    /* ── Header ──────────────────────────────────────────────────── */
    .main-header {
        text-align: center;
        padding: 2rem 0 1rem;
        margin-bottom: 1.5rem;
    }
    .main-header h1 {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #0284C7, #0369A1, #0EA5E9);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.3rem;
        letter-spacing: -0.5px;
    }
    .main-header p {
        color: #475569;
        font-size: 1rem;
        font-weight: 400;
    }

    /* ── Glass card ──────────────────────────────────────────────── */
    .glass-card {
        background: rgba(255, 255, 255, 0.85);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(2, 132, 199, 0.15);
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 12px rgba(2, 132, 199, 0.05);
        transition: all 0.3s ease;
    }
    .glass-card:hover {
        border-color: rgba(2, 132, 199, 0.4);
        box-shadow: 0 8px 24px rgba(2, 132, 199, 0.1);
    }

    /* ── Sentiment result ────────────────────────────────────────── */
    .sentiment-result {
        text-align: center;
        padding: 2rem;
        border-radius: 16px;
        margin: 1rem 0;
        animation: fadeInUp 0.5s ease;
    }
    .sentiment-label {
        font-size: 2rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-bottom: 0.5rem;
    }
    .sentiment-emoji {
        font-size: 3rem;
        margin-bottom: 0.5rem;
        display: block;
    }
    .confidence-badge {
        display: inline-block;
        padding: 0.3rem 1rem;
        border-radius: 20px;
        background: rgba(255,255,255,0.25);
        font-size: 0.9rem;
        color: #FFFFFF;
        font-weight: 500;
    }

    /* ── Reason card ─────────────────────────────────────────────── */
    .reason-card {
        background: rgba(241, 245, 249, 0.9);
        border-left: 3px solid #0284C7;
        padding: 0.7rem 1rem;
        margin: 0.4rem 0;
        border-radius: 0 8px 8px 0;
        font-size: 0.9rem;
        color: #1E293B;
    }

    /* ── Word tags ───────────────────────────────────────────────── */
    .word-tag {
        display: inline-block;
        padding: 0.25rem 0.7rem;
        margin: 0.2rem;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 500;
    }
    .word-positive { background: rgba(34, 197, 94, 0.15); color: #16A34A; border: 1px solid rgba(34, 197, 94, 0.25); }
    .word-negative { background: rgba(239, 68, 68, 0.12); color: #DC2626; border: 1px solid rgba(239, 68, 68, 0.22); }
    .word-neutral  { background: rgba(245, 158, 11, 0.15); color: #D97706; border: 1px solid rgba(245, 158, 11, 0.25); }

    /* ── Metric card ─────────────────────────────────────────────── */
    .metric-card {
        background: rgba(255, 255, 255, 0.9);
        border: 1px solid rgba(2, 132, 199, 0.2);
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        box-shadow: 0 2px 8px rgba(2, 132, 199, 0.03);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #0284C7, #0EA5E9);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-label {
        color: #475569;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 0.3rem;
    }

    /* ── History item ────────────────────────────────────────────── */
    .history-item {
        background: rgba(255, 255, 255, 0.7);
        border: 1px solid rgba(2, 132, 199, 0.15);
        border-radius: 10px;
        padding: 0.8rem;
        margin-bottom: 0.5rem;
        box-shadow: 0 2px 6px rgba(0,0,0,0.02);
    }
    .history-text {
        color: #1E293B;
        font-size: 0.85rem;
        margin-bottom: 0.3rem;
    }

    /* ── Tab styling ─────────────────────────────────────────────── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(224, 242, 254, 0.5);
        border-radius: 12px;
        padding: 4px;
        border: 1px solid rgba(2, 132, 199, 0.1);
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        color: #475569;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background: rgba(2, 132, 199, 0.15) !important;
        color: #0284C7 !important;
    }

    /* ── Button styling ──────────────────────────────────────────── */
    .stButton > button {
        background: linear-gradient(135deg, #0284C7, #0369A1) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.6rem 2rem !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        transition: all 0.3s ease !important;
        width: 100%;
        box-shadow: 0 4px 12px rgba(2, 132, 199, 0.2) !important;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #0369A1, #075985) !important;
        box-shadow: 0 4px 20px rgba(2, 132, 199, 0.4) !important;
        transform: translateY(-1px) !important;
    }

    /* ── Text area styling ───────────────────────────────────────── */
    .stTextArea textarea {
        background: rgba(255, 255, 255, 0.95) !important;
        border: 1px solid rgba(2, 132, 199, 0.2) !important;
        border-radius: 12px !important;
        color: #0F172A !important;
        font-size: 1rem !important;
    }
    .stTextArea textarea:focus {
        border-color: #0284C7 !important;
        box-shadow: 0 0 0 2px rgba(2, 132, 199, 0.2) !important;
    }

    /* ── Animation ───────────────────────────────────────────────── */
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(20px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    .animate-in { animation: fadeInUp 0.5s ease; }

    /* ── Sidebar ─────────────────────────────────────────────────── */
    section[data-testid="stSidebar"] {
        background: rgba(241, 245, 249, 0.98) !important;
        border-right: 1px solid rgba(2, 132, 199, 0.1);
    }

    /* ── Hide default Streamlit elements ─────────────────────────── */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

if "history" not in st.session_state:
    st.session_state.history = []


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

@st.cache_resource
def load_model():
    """Load saved model and vectorizer."""
    model_path = os.path.join(MODELS_DIR, "best_model.joblib")
    vec_path = os.path.join(MODELS_DIR, "tfidf_vectorizer.joblib")
    report_path = os.path.join(MODELS_DIR, "training_report.json")

    if not os.path.exists(model_path):
        return None, None, None

    model = joblib.load(model_path)
    vectorizer = joblib.load(vec_path)

    report = None
    if os.path.exists(report_path):
        with open(report_path, "r") as f:
            report = json.load(f)

    ensure_nltk_data()
    return model, vectorizer, report


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------

def create_confidence_gauge(probabilities):
    """Create a horizontal bar chart showing sentiment probabilities."""
    if not probabilities:
        return None

    sentiments = list(probabilities.keys())
    values = [probabilities[s] * 100 for s in sentiments]
    colors = [SENTIMENT_CONFIG[s]["color"] for s in sentiments]

    fig = go.Figure()
    for i, (s, v, c) in enumerate(zip(sentiments, values, colors)):
        fig.add_trace(go.Bar(
            y=[s.capitalize()],
            x=[v],
            orientation="h",
            marker=dict(color=c),
            text=f"{v:.1f}%",
            textposition="outside",
            textfont=dict(color=c, size=14, family="Inter"),
            name=s.capitalize(),
            showlegend=False,
        ))

    fig.update_layout(
        xaxis=dict(range=[0, 105], showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, tickfont=dict(size=13, color="#94A3B8", family="Inter")),
        height=160,
        margin=dict(l=0, r=40, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        bargap=0.35,
    )
    return fig





# ---------------------------------------------------------------------------
# UI Components
# ---------------------------------------------------------------------------

def render_header():
    st.markdown("""
    <div class="main-header">
        <h1>🔮 Sentiment Analyzer</h1>
        <p>AI-powered sentiment analysis with explainable predictions</p>
    </div>
    """, unsafe_allow_html=True)


def render_sentiment_result(explanation):
    """Render the full sentiment analysis result with explanation."""
    sentiment = explanation["predicted_sentiment"]
    config = SENTIMENT_CONFIG[sentiment]
    confidence = explanation.get("confidence")

    # Main result card
    st.markdown(f"""
    <div class="sentiment-result" style="background: {config['gradient']}; opacity: 0.95;">
        <span class="sentiment-emoji">{config['emoji']}</span>
        <div class="sentiment-label" style="color: white;">{sentiment}</div>
        {f'<span class="confidence-badge">Confidence: {confidence:.1%}</span>' if confidence else ''}
    </div>
    """, unsafe_allow_html=True)

    # Probability breakdown
    if explanation.get("probabilities"):
        st.markdown("##### 📊 Sentiment Probability Breakdown")
        fig = create_confidence_gauge(explanation["probabilities"])
        if fig:
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # Key reasons
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("##### 🔍 Key Reasons")
        if explanation.get("key_reasons"):
            for reason in explanation["key_reasons"]:
                st.markdown(f'<div class="reason-card">• {reason}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="reason-card">No strong sentiment indicators found.</div>',
                        unsafe_allow_html=True)

    with col2:
        st.markdown("##### 🏷️ Word Sentiment Map")
        # Positive words
        if explanation.get("positive_words"):
            pos_tags = " ".join(
                f'<span class="word-tag word-positive">{w}</span>'
                for w in explanation["positive_words"][:8]
            )
            st.markdown(f"**Positive:** {pos_tags}", unsafe_allow_html=True)

        # Negative words
        if explanation.get("negative_words"):
            neg_tags = " ".join(
                f'<span class="word-tag word-negative">{w}</span>'
                for w in explanation["negative_words"][:8]
            )
            st.markdown(f"**Negative:** {neg_tags}", unsafe_allow_html=True)

        # Neutral words
        if explanation.get("neutral_words"):
            neu_tags = " ".join(
                f'<span class="word-tag word-neutral">{w}</span>'
                for w in explanation["neutral_words"][:8]
            )
            st.markdown(f"**Neutral:** {neu_tags}", unsafe_allow_html=True)

    # Key phrases
    if explanation.get("key_phrases"):
        st.markdown("##### 💡 Extracted Key Phrases")
        phrase_tags = " ".join(
            f'<span class="word-tag word-neutral">{p}</span>'
            for p in explanation["key_phrases"]
        )
        st.markdown(phrase_tags, unsafe_allow_html=True)


def render_history():
    """Render analysis history in sidebar."""
    st.sidebar.markdown("### 📋 Recent Analyses")
    if not st.session_state.history:
        st.sidebar.markdown(
            '<p style="color: #64748B; font-size: 0.85rem;">No analyses yet. '
            'Try analyzing some text!</p>',
            unsafe_allow_html=True,
        )
    else:
        for item in reversed(st.session_state.history[-10:]):
            config = SENTIMENT_CONFIG[item["sentiment"]]
            display_text = item["text"][:60] + ("..." if len(item["text"]) > 60 else "")
            confidence_str = f" ({item['confidence']:.0%})" if item.get("confidence") else ""
            st.sidebar.markdown(
                f"""<div class="history-item">
                    <div class="history-text">{display_text}</div>
                    <span class="word-tag word-{item['sentiment']}">{config['emoji']} {item['sentiment'].capitalize()}{confidence_str}</span>
                </div>""",
                unsafe_allow_html=True,
            )


# ---------------------------------------------------------------------------
# Tab: Analyze
# ---------------------------------------------------------------------------

def tab_analyze(model, vectorizer):
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("#### ✍️ Enter Text to Analyze")

    text_input = st.text_area(
        "Paste a customer review, tweet, or any sentence:",
        height=130,
        placeholder="e.g., 'The product quality is amazing! Fast delivery and great packaging. Highly recommend!'",
        label_visibility="collapsed",
    )

    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        analyze_clicked = st.button("🔍 Analyze Sentiment", use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # Example sentences
    st.markdown("##### 💡 Try these examples:")
    example_cols = st.columns(3)
    examples = [
        "This product is absolutely amazing! Best purchase ever!",
        "Terrible service. I waited 3 hours and nobody helped me.",
        "The package arrived on time. It's an average product.",
    ]
    example_labels = ["Positive Example", "Negative Example", "Neutral Example"]

    selected_example = None
    for col, example, label in zip(example_cols, examples, example_labels):
        with col:
            if st.button(f"📝 {label}", key=f"ex_{label}", use_container_width=True):
                selected_example = example

    # Use selected example if clicked
    active_text = selected_example if selected_example else text_input

    if (analyze_clicked and text_input.strip()) or selected_example:
        if not active_text.strip():
            st.warning("⚠️ Please enter some text to analyze.")
            return

        with st.spinner("🔮 Analyzing sentiment..."):
            explanation = explain_prediction(active_text, vectorizer, model)

        render_sentiment_result(explanation)

        # Save to history
        st.session_state.history.append({
            "text": active_text,
            "sentiment": explanation["predicted_sentiment"],
            "confidence": explanation.get("confidence"),
        })


# ---------------------------------------------------------------------------
# Tab: Batch Analysis
# ---------------------------------------------------------------------------

def tab_batch(model, vectorizer):
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("#### 📁 Upload CSV for Batch Analysis")
    st.markdown(
        '<p style="color: #94A3B8; font-size: 0.9rem;">'
        'Upload a CSV file with a column named <b>text</b> or <b>review</b> containing the sentences to analyze.</p>',
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"], label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        # Find text column
        text_col = None
        for col_name in ["text", "review", "sentence", "comment", "content", "Text", "Review"]:
            if col_name in df.columns:
                text_col = col_name
                break

        if text_col is None:
            st.error("❌ Could not find a text column. Please ensure your CSV has a column named 'text' or 'review'.")
            return

        st.markdown(f"**Found text column:** `{text_col}` — **{len(df)} rows**")

        if st.button("🚀 Run Batch Analysis", use_container_width=True):
            progress = st.progress(0, text="Analyzing...")
            results = []

            for i, row in df.iterrows():
                text = str(row[text_col])
                cleaned = clean_text(text)
                vec = vectorizer.transform([cleaned])
                pred = model.predict(vec)[0]
                label = LABEL_NAMES[pred]

                confidence = None
                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba(vec)[0]
                    confidence = float(max(proba))

                results.append({
                    "text": text[:100],
                    "sentiment": label,
                    "confidence": f"{confidence:.1%}" if confidence else "N/A",
                })

                progress.progress((i + 1) / len(df), text=f"Analyzing... ({i + 1}/{len(df)})")

            progress.empty()

            results_df = pd.DataFrame(results)

            # Summary metrics
            st.markdown("##### 📊 Batch Results Summary")
            m_cols = st.columns(4)
            total = len(results_df)
            pos_count = len(results_df[results_df["sentiment"] == "positive"])
            neg_count = len(results_df[results_df["sentiment"] == "negative"])
            neu_count = len(results_df[results_df["sentiment"] == "neutral"])

            with m_cols[0]:
                st.markdown(f"""<div class="metric-card">
                    <div class="metric-value">{total}</div>
                    <div class="metric-label">Total Reviews</div>
                </div>""", unsafe_allow_html=True)
            with m_cols[1]:
                st.markdown(f"""<div class="metric-card">
                    <div class="metric-value" style="-webkit-text-fill-color: #22C55E;">{pos_count}</div>
                    <div class="metric-label">Positive</div>
                </div>""", unsafe_allow_html=True)
            with m_cols[2]:
                st.markdown(f"""<div class="metric-card">
                    <div class="metric-value" style="-webkit-text-fill-color: #EF4444;">{neg_count}</div>
                    <div class="metric-label">Negative</div>
                </div>""", unsafe_allow_html=True)
            with m_cols[3]:
                st.markdown(f"""<div class="metric-card">
                    <div class="metric-value" style="-webkit-text-fill-color: #F59E0B;">{neu_count}</div>
                    <div class="metric-label">Neutral</div>
                </div>""", unsafe_allow_html=True)

            # Distribution pie chart
            fig = go.Figure(data=[go.Pie(
                labels=["Positive", "Negative", "Neutral"],
                values=[pos_count, neg_count, neu_count],
                hole=0.5,
                marker=dict(colors=["#22C55E", "#EF4444", "#F59E0B"]),
                textfont=dict(color="white", size=13),
            )])
            fig.update_layout(
                height=300,
                margin=dict(l=20, r=20, t=20, b=20),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                legend=dict(font=dict(color="#94A3B8", size=12)),
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

            # Results table
            st.markdown("##### 📋 Detailed Results")
            st.dataframe(results_df, use_container_width=True, height=400)

            # Download button
            csv_data = results_df.to_csv(index=False)
            st.download_button(
                "📥 Download Results CSV",
                csv_data,
                file_name="sentiment_results.csv",
                mime="text/csv",
                use_container_width=True,
            )


# ---------------------------------------------------------------------------
# Tab: Model Performance
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    render_header()

    model, vectorizer, report = load_model()

    if model is None:
        st.error(
            "⚠️ **No trained model found!**\n\n"
            "Please run the training pipeline first:\n"
            "```bash\npython train_models.py\n```"
        )
        return

    # Sidebar
    st.sidebar.markdown("""
    <div style="text-align: center; padding: 1rem 0;">
        <h2 style="background: linear-gradient(135deg, #7C3AED, #A78BFA);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
            font-size: 1.3rem; font-weight: 700;">🔮 Sentiment Analyzer</h2>
    </div>
    """, unsafe_allow_html=True)

    if report:
        st.sidebar.markdown(f"""
        <div class="glass-card" style="padding: 0.8rem;">
            <p style="color: #94A3B8; font-size: 0.8rem; margin: 0;">
                <b style="color: #A78BFA;">Active Model</b><br>
                {report['best_model_name']}<br>
                <b style="color: #22C55E;">F1: {report['best_f1_score']:.1%}</b> •
                <b style="color: #A78BFA;">Acc: {report['best_accuracy']:.1%}</b>
            </p>
        </div>
        """, unsafe_allow_html=True)

    render_history()

    # Tabs
    tab1, tab2 = st.tabs(["🎯 Analyze", "📁 Batch Analysis"])

    with tab1:
        tab_analyze(model, vectorizer)
    with tab2:
        tab_batch(model, vectorizer)


if __name__ == "__main__":
    main()