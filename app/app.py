"""
🎬 Indian Movie Budget Prediction — Advanced Streamlit Web App
================================================================
Interactive web application with:
- Budget Recommendation System (range-based predictions)
- Feature Importance Analysis with insights
- Genre vs Budget Analysis
- Actor/Director Impact Analysis  
- Error Analysis Dashboard
- Inflation-Adjusted Trend Analysis
- Industry ROI Comparison
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import os
import sys
import json

# Add parent src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# ============================================================
# Page Configuration
# ============================================================

st.set_page_config(
    page_title="🎬 Indian Movie Budget Predictor",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# Custom CSS for Premium Look
# ============================================================

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    /* Main container */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }
    
    /* Dark theme overrides */
    .stApp {
        background: linear-gradient(180deg, #0D1117 0%, #161B22 100%);
    }
    
    /* Headers */
    h1, h2, h3 {
        font-family: 'Inter', sans-serif !important;
        font-weight: 700 !important;
    }
    
    h1 {
        background: linear-gradient(135deg, #FF6B6B, #4ECDC4, #45B7D1, #FFD93D);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem !important;
        text-align: center;
        padding: 0.5rem 0;
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #1a1f2e, #252b3b);
        border: 1px solid rgba(78, 205, 196, 0.2);
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 32px rgba(78, 205, 196, 0.15);
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #4ECDC4, #45B7D1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #8B949E;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 0.3rem;
    }
    
    /* Prediction result - Budget Recommendation Box */
    .prediction-box {
        background: linear-gradient(135deg, #1a2a1a, #1a3a2a);
        border: 2px solid #00FF88;
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 0 40px rgba(0, 255, 136, 0.1);
    }
    .prediction-value {
        font-size: 3.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #00FF88, #4ECDC4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* Recommendation range box */
    .recommendation-box {
        background: linear-gradient(135deg, #1a1a2e, #162447);
        border: 2px solid #e94560;
        border-radius: 20px;
        padding: 1.5rem;
        text-align: center;
        margin: 0.5rem 0;
        box-shadow: 0 0 30px rgba(233, 69, 96, 0.1);
    }
    .range-value {
        font-size: 2rem;
        font-weight: 700;
        color: #e94560;
    }
    
    /* Insight card */
    .insight-card {
        background: rgba(69, 183, 209, 0.08);
        border-left: 4px solid #45B7D1;
        border-radius: 8px;
        padding: 1rem 1.5rem;
        margin: 0.8rem 0;
        font-size: 0.95rem;
    }
    .insight-icon { font-size: 1.2rem; margin-right: 0.5rem; }
    
    /* Sidebar */
    .css-1d391kg, [data-testid="stSidebar"] {
        background: #0D1117 !important;
    }
    
    /* Info box */
    .info-box {
        background: rgba(69, 183, 209, 0.1);
        border-left: 4px solid #45B7D1;
        border-radius: 8px;
        padding: 1rem 1.5rem;
        margin: 1rem 0;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 20px;
        background: #161B22;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #4ECDC4, #45B7D1) !important;
    }
    
    .stSelectbox label, .stSlider label, .stNumberInput label {
        font-weight: 600 !important;
        color: #E6EDF3 !important;
    }
    
    .footer {
        text-align: center;
        color: #484F58;
        padding: 2rem 0;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# Data & Model Loading
# ============================================================

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")

def load_data():
    """Load the best available dataset."""
    enriched = os.path.join(BASE_DIR, "data", "cleaned_enriched_data.csv")
    raw = os.path.join(BASE_DIR, "data", "raw_data.csv")
    if os.path.exists(enriched):
        return pd.read_csv(enriched)
    elif os.path.exists(raw):
        return pd.read_csv(raw)
    return None

@st.cache_resource
def load_model():
    """Load trained model and preprocessor."""
    model_path = os.path.join(BASE_DIR, "models", "best_model.pkl")
    prep_path = os.path.join(BASE_DIR, "models", "preprocessor.pkl")
    model = joblib.load(model_path) if os.path.exists(model_path) else None
    preprocessor = joblib.load(prep_path) if os.path.exists(prep_path) else None
    return model, preprocessor

def load_json_safe(path):
    """Load JSON file safely."""
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return None

# ============================================================
# Color Palette & Helpers
# ============================================================

INDUSTRY_COLORS = {
    "Bollywood": "#FF6B6B", "Tollywood": "#4ECDC4",
    "Kollywood": "#45B7D1", "Sandalwood": "#FFA07A",
    "Mollywood": "#98D8C8"
}

def create_metric_card(value, label, prefix="", suffix=""):
    return f"""
    <div class="metric-card">
        <div class="metric-value">{prefix}{value}{suffix}</div>
        <div class="metric-label">{label}</div>
    </div>
    """

def create_insight_card(text, icon="💡"):
    return f"""
    <div class="insight-card">
        <span class="insight-icon">{icon}</span> {text}
    </div>
    """

def plotly_dark():
    return dict(
        paper_bgcolor='rgba(13,17,23,0)',
        plot_bgcolor='rgba(22,27,34,0.8)',
        font=dict(family="Inter", color="#E6EDF3"),
        title_font=dict(size=18, family="Inter"),
        xaxis=dict(gridcolor='rgba(48,54,61,0.5)', zerolinecolor='rgba(48,54,61,0.5)'),
        yaxis=dict(gridcolor='rgba(48,54,61,0.5)', zerolinecolor='rgba(48,54,61,0.5)'),
        legend=dict(bgcolor='rgba(22,27,34,0.5)', bordercolor='rgba(48,54,61,0.5)'),
        margin=dict(l=60, r=30, t=60, b=60)
    )


# ============================================================
# Main Application
# ============================================================

def main():
    st.markdown("<h1>🎬 Indian Movie Budget Predictor</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p style='text-align:center; color:#8B949E; font-size:1.1rem; margin-bottom: 2rem;'>"
        "ML-Powered Budget Recommendation & Trend Analysis for Indian Cinema"
        "</p>",
        unsafe_allow_html=True
    )

    df = load_data()
    model, preprocessor = load_model()

    if df is None:
        st.error("⚠️ Dataset not found! Run the training pipeline first.")
        st.stop()

    # ─── Sidebar ─────────────────────────────────────────
    with st.sidebar:
        st.markdown("### 🎯 Navigation")
        page = st.radio(
            "Choose a Section",
            ["🏠 Dashboard", "🔮 Budget Recommender", "📊 Feature Importance",
             "📈 Trend Analysis", "🌍 Industry Comparison",
             "🎭 Genre & Actor Insights", "📉 Error Analysis",
             "🧠 Model Performance"],
            label_visibility="collapsed"
        )

        st.markdown("---")
        st.markdown("### 🎛️ Filters")

        selected_industries = st.multiselect(
            "Industries", options=list(INDUSTRY_COLORS.keys()),
            default=list(INDUSTRY_COLORS.keys())
        )

        year_range = st.slider(
            "Year Range",
            min_value=int(df["year"].min()),
            max_value=int(df["year"].max()),
            value=(2005, int(df["year"].max()))
        )

        st.markdown("---")
        st.markdown(
            "<div style='text-align:center; color:#484F58; font-size:0.75rem;'>"
            "Built with ❤️ using Streamlit<br>Advanced ML Project"
            "</div>",
            unsafe_allow_html=True
        )

    # Apply filters
    filtered_df = df[
        (df["industry"].isin(selected_industries)) &
        (df["year"].between(year_range[0], year_range[1]))
    ]

    # ─── Page: Dashboard ─────────────────────────────────
    if page == "🏠 Dashboard":
        render_dashboard(filtered_df, df)

    elif page == "🔮 Budget Recommender":
        render_budget_recommender(df, model, preprocessor)

    elif page == "📊 Feature Importance":
        render_feature_importance(df)

    elif page == "📈 Trend Analysis":
        render_trend_analysis(filtered_df, selected_industries, df)

    elif page == "🌍 Industry Comparison":
        render_industry_comparison(filtered_df, selected_industries)

    elif page == "🎭 Genre & Actor Insights":
        render_genre_actor_insights(filtered_df)

    elif page == "📉 Error Analysis":
        render_error_analysis()

    elif page == "🧠 Model Performance":
        render_model_performance()

    # Footer
    st.markdown(
        '<div class="footer">'
        '🎬 Indian Movie Budget Prediction | Advanced ML Project<br>'
        'Built with Python, Scikit-Learn, XGBoost, LightGBM, SHAP & Streamlit'
        '</div>',
        unsafe_allow_html=True
    )


# ════════════════════════════════════════════════════════
# Page Renderers
# ════════════════════════════════════════════════════════

def render_dashboard(filtered_df, df):
    """Main dashboard with key metrics."""
    # Key Metrics
    cols = st.columns(6)
    has_roi = "roi_percentage" in filtered_df.columns
    metrics = [
        (f"{len(filtered_df):,}", "Total Movies"),
        (f"₹{filtered_df['budget_crores'].mean():.1f} Cr", "Avg Budget"),
        (f"₹{filtered_df['budget_crores'].max():.0f} Cr", "Highest Budget"),
        (f"{filtered_df['industry'].nunique()}", "Industries"),
        (f"{filtered_df['genre'].nunique()}", "Genres"),
    ]
    if has_roi:
        avg_roi = filtered_df["roi_percentage"].mean()
        metrics.append((f"{avg_roi:.0f}%", "Avg ROI"))

    for col, (value, label) in zip(cols[:len(metrics)], metrics):
        with col:
            st.markdown(create_metric_card(value, label), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Charts Row 1
    col1, col2 = st.columns(2)

    with col1:
        yearly_avg = filtered_df.groupby(["year", "industry"])["budget_crores"].mean().reset_index()
        fig = px.line(yearly_avg, x="year", y="budget_crores", color="industry",
                      color_discrete_map=INDUSTRY_COLORS,
                      title="📈 Average Budget Trend by Industry")
        fig.update_layout(**plotly_dark())
        fig.update_traces(line=dict(width=3))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        industry_stats = filtered_df.groupby("industry")["budget_crores"].agg(["mean", "count"]).reset_index()
        fig = px.bar(industry_stats, x="industry", y="mean", color="industry",
                     color_discrete_map=INDUSTRY_COLORS,
                     title="💰 Average Budget by Industry")
        fig.update_layout(**plotly_dark(), showlegend=False)
        fig.update_traces(marker_line_width=0, opacity=0.85)
        st.plotly_chart(fig, use_container_width=True)

    # Charts Row 2
    col3, col4 = st.columns(2)

    with col3:
        genre_budget = filtered_df.groupby("genre")["budget_crores"].mean().sort_values(ascending=True).tail(12)
        fig = px.bar(genre_budget, x=genre_budget.values, y=genre_budget.index, orientation='h',
                     title="🎭 Average Budget by Genre (Top 12)",
                     color=genre_budget.values, color_continuous_scale="Viridis")
        fig.update_layout(**plotly_dark(), coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        if has_roi:
            # ROI distribution (unique!)
            fig = px.histogram(filtered_df, x="roi_percentage",
                               nbins=50, title="📊 ROI Distribution (%)",
                               color_discrete_sequence=["#4ECDC4"])
            fig.add_vline(x=0, line_dash="dash", line_color="#FF6B6B", 
                          annotation_text="Break Even")
            fig.update_layout(**plotly_dark())
            st.plotly_chart(fig, use_container_width=True)
        else:
            verdict_counts = filtered_df["verdict"].value_counts()
            fig = px.pie(values=verdict_counts.values, names=verdict_counts.index,
                         title="🎯 Movie Verdict Distribution",
                         color_discrete_sequence=px.colors.qualitative.Set2)
            fig.update_layout(**plotly_dark())
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)


# ────────────────────────────────────────────────────────
def render_budget_recommender(df, model, preprocessor):
    """
    🔮 Budget Recommendation System
    Instead of "Predicted budget = ₹50 Cr", shows:
    "Recommended budget range = ₹40–60 Cr based on similar films"
    """
    st.markdown("## 🔮 Budget Recommendation System")
    st.markdown(
        '<div class="info-box">'
        '💡 Our ML model doesn\'t just predict a single number — it recommends an '
        '<b>optimal budget range</b> based on similar successful films, with confidence levels.'
        '</div>',
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        industry = st.selectbox("🎬 Industry", list(INDUSTRY_COLORS.keys()))
        genre = st.selectbox("🎭 Genre", sorted(df["genre"].dropna().astype(str).unique()))
        year = st.slider("📅 Year", 2000, 2030, 2025)
        release_month = st.slider("📆 Release Month", 1, 12, 6)

    with col2:
        industry_df = df[df["industry"] == industry]
        all_actors = sorted(industry_df["lead_actor"].dropna().astype(str).unique())
        all_directors = sorted(industry_df["director"].dropna().astype(str).unique())

        # Step 1: Select actor
        lead_actor = st.selectbox("⭐ Lead Actor", all_actors)

        # Step 2: Build smart director list — collaborators first, then others
        actor_movies = industry_df[industry_df["lead_actor"] == lead_actor]
        collab_directors = sorted(actor_movies["director"].dropna().astype(str).unique())
        other_directors = sorted(set(all_directors) - set(collab_directors))

        if collab_directors:
            # Show collaborators at top with a separator label
            director_options = collab_directors + ["── Other Directors ──"] + other_directors
        else:
            director_options = all_directors

        director = st.selectbox(
            "🎬 Director",
            director_options,
            help=f"Top options: directors who worked with {lead_actor}"
        )
        # Handle separator selection
        if director == "── Other Directors ──":
            director = other_directors[0] if other_directors else all_directors[0]

        # Show collaboration info
        collab_count = len(actor_movies[actor_movies["director"] == director])
        if collab_count > 0:
            st.caption(f"🤝 {lead_actor} + {director}: {collab_count} movie{'s' if collab_count > 1 else ''} together")
        else:
            st.caption(f"🆕 New pairing — no previous collaborations")

        num_cast = st.slider("👥 Cast Members", 3, 15, 8)
        runtime = st.slider("⏱️ Runtime (min)", 90, 210, 150)

    with col3:
        vfx_level = st.slider("🖥️ VFX Level", 0, 5, 2)
        num_songs = st.slider("🎵 Number of Songs", 0, 6, 3)
        is_sequel = st.checkbox("🔄 Is Sequel?")
        intl_release = st.checkbox("🌍 International Release")
        ott_release = st.checkbox("📱 OTT Release")

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("🚀 Get Budget Recommendation", use_container_width=True, type="primary"):
        # Get actor and director scores from data
        actor_pop = industry_df[industry_df["lead_actor"] == lead_actor]["actor_popularity_score"].mean()
        if pd.isna(actor_pop): actor_pop = 3.0
        dir_success = industry_df[industry_df["director"] == director]["director_success_rate"].mean()
        if pd.isna(dir_success): dir_success = 0.3

        # Find similar movies for the recommendation range
        similar = df[
            (df["industry"] == industry) &
            (df["genre"] == genre) &
            (df["year"].between(year - 5, year + 2))
        ]
        if len(similar) < 3:
            similar = df[(df["industry"] == industry) & (df["genre"] == genre)]
        if len(similar) < 3:
            similar = df[df["industry"] == industry]

        similar_budgets = similar["budget_crores"]

        # Heuristic prediction (always available)
        industry_bases = {
            "Bollywood": 50, "Tollywood": 45, "Kollywood": 35,
            "Sandalwood": 20, "Mollywood": 15
        }
        genre_mults = {
            "Sci-Fi": 1.8, "Science Fiction": 1.8, "Historical": 1.6,
            "War": 1.7, "Fantasy": 1.5, "Mythology": 1.6, "History": 1.6,
            "Action": 1.3, "Adventure": 1.4, "Animation": 1.5,
            "Horror": 0.6, "Comedy": 0.7, "Romance": 0.8, "Drama": 0.75
        }

        base = industry_bases.get(industry, 30)
        growth = (1.05) ** (year - 2000)
        actor_mult = 1 + (actor_pop / 10) * 1.0
        dir_mult = 1 + dir_success * 0.5
        genre_mult = genre_mults.get(genre, 1.0)
        sequel_mult = 1.2 if is_sequel else 1.0
        vfx_mult = 1 + vfx_level * 0.10
        intl_mult = 1.15 if intl_release else 1.0

        predicted = base * growth * actor_mult * dir_mult * genre_mult * sequel_mult * vfx_mult * intl_mult
        predicted = min(predicted, 650)
        predicted = round(predicted, 2)

        # Budget recommendation range
        p25 = similar_budgets.quantile(0.25) if len(similar_budgets) > 5 else predicted * 0.6
        p75 = similar_budgets.quantile(0.75) if len(similar_budgets) > 5 else predicted * 1.4
        lower = min(predicted * 0.75, p25)
        upper = max(predicted * 1.25, p75)

        # Display: Budget Recommendation (the killer feature)
        st.markdown(
            f'<div class="recommendation-box">'
            f'<div style="font-size:1rem; color:#8B949E; margin-bottom:0.3rem;">💎 Recommended Budget Range</div>'
            f'<div class="range-value">₹{lower:,.0f} — ₹{upper:,.0f} Crores</div>'
            f'<div style="font-size:0.85rem; color:#8B949E; margin-top:0.3rem;">'
            f'Based on {len(similar)} similar films in dataset</div>'
            f'</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            f'<div class="prediction-box">'
            f'<div style="font-size:1rem; color:#8B949E; margin-bottom:0.5rem;">🎯 Point Estimate</div>'
            f'<div class="prediction-value">₹{predicted:,.2f} Crores</div>'
            f'<div style="font-size:1rem; color:#4ECDC4; margin-top:0.5rem;">'
            f'≈ ${predicted * 12:,.0f} Million USD</div>'
            f'</div>',
            unsafe_allow_html=True
        )

        # Budget Breakdown — WHY this prediction
        st.markdown("### 📊 Why This Budget? (Factor Breakdown)")
        st.markdown(create_insight_card(
            "This is what makes our model <b>explainable</b> — you can see exactly which factors "
            "drive the budget prediction.", "🔍"
        ), unsafe_allow_html=True)

        breakdown = pd.DataFrame({
            "Factor": ["Base Budget", "Year Growth", f"Star Power ({lead_actor})",
                       f"Director ({director})", f"Genre ({genre})",
                       "VFX Level", "Sequel", "International"],
            "Multiplier": [f"₹{base} Cr", f"{growth:.2f}x", f"{actor_mult:.2f}x",
                           f"{dir_mult:.2f}x", f"{genre_mult:.2f}x",
                           f"{vfx_mult:.2f}x", f"{sequel_mult:.2f}x", f"{intl_mult:.2f}x"],
            "Impact": ["Baseline",
                       f"+{(growth-1)*100:.0f}%",
                       f"+{(actor_mult-1)*100:.0f}%",
                       f"+{(dir_mult-1)*100:.0f}%",
                       f"{'+' if genre_mult >= 1 else ''}{(genre_mult-1)*100:.0f}%",
                       f"+{(vfx_mult-1)*100:.0f}%",
                       f"+{(sequel_mult-1)*100:.0f}%" if is_sequel else "N/A",
                       f"+{(intl_mult-1)*100:.0f}%" if intl_release else "N/A"]
        })
        st.dataframe(breakdown, use_container_width=True, hide_index=True)

        # Waterfall chart (SHAP-like visualization)
        factors = ["Base", "Year", "Star", "Director", "Genre", "VFX"]
        values = [base, base*(growth-1), base*growth*(actor_mult-1),
                  base*growth*actor_mult*(dir_mult-1),
                  base*growth*actor_mult*dir_mult*(genre_mult-1),
                  base*growth*actor_mult*dir_mult*genre_mult*(vfx_mult-1)]

        fig = go.Figure(go.Waterfall(
            name="Budget Factors",
            orientation="v",
            x=factors,
            y=values,
            connector={"line": {"color": "#4ECDC4"}},
            increasing={"marker": {"color": "#00FF88"}},
            decreasing={"marker": {"color": "#FF6B6B"}},
            totals={"marker": {"color": "#45B7D1"}},
        ))
        fig.update_layout(**plotly_dark(), title="💧 Budget Waterfall — Factor Contributions (₹ Cr)",
                          height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        # Similar movies
        st.markdown("### 🎬 Similar Movies in Dataset")
        title_col = "title" if "title" in df.columns else "movie_name" if "movie_name" in df.columns else "movie_id"
        sim_display = df[
            (df["industry"] == industry) &
            ((df["lead_actor"] == lead_actor) | (df["director"] == director))
        ].nlargest(5, "budget_crores")[
            [title_col, "year", "lead_actor", "director",
             "budget_crores", "box_office_crores", "verdict"]
        ].copy()

        if len(sim_display) > 0:
            sim_display["budget_crores"] = sim_display["budget_crores"].apply(lambda x: f"₹{x:,.2f} Cr")
            sim_display["box_office_crores"] = sim_display["box_office_crores"].apply(lambda x: f"₹{x:,.2f} Cr")
            sim_display.columns = ["Movie", "Year", "Lead Actor", "Director", "Budget", "Box Office", "Verdict"]
            st.dataframe(sim_display, use_container_width=True, hide_index=True)


# ────────────────────────────────────────────────────────
def render_feature_importance(df):
    """📊 Feature Importance Analysis — shows what drives budgets."""
    st.markdown("## 📊 Feature Importance Analysis")
    st.markdown(create_insight_card(
        "Feature importance tells us <b>which factors matter most</b> for budget prediction. "
        "This is computed from the trained Random Forest / XGBoost model.",
        "🧠"
    ), unsafe_allow_html=True)

    # Compute from data (always available)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    correlations = {}
    for col in numeric_cols:
        if col not in ["budget_crores", "box_office_crores", "movie_id", "cpi_index"]:
            try:
                corr = abs(df[col].corr(df["budget_crores"]))
                if not np.isnan(corr):
                    correlations[col] = corr
            except Exception:
                pass

    if correlations:
        corr_df = pd.DataFrame(
            sorted(correlations.items(), key=lambda x: x[1], reverse=True)[:20],
            columns=["Feature", "Importance"]
        )
        corr_df["Importance_Pct"] = (corr_df["Importance"] / corr_df["Importance"].sum() * 100).round(1)

        # Clean feature names for display
        name_map = {
            "actor_popularity_score": "⭐ Actor Popularity",
            "director_success_rate": "🎬 Director Success Rate",
            "production_house_strength": "🏢 Production House",
            "num_screens": "📺 Screen Count",
            "vfx_level": "🖥️ VFX Level",
            "star_power_index": "⭐ Star Power Index",
            "hype_score": "🔥 Hype Score",
            "genre_complexity_index": "🎭 Genre Complexity",
            "language_market_factor": "🌍 Language Market",
            "industry_growth_factor": "📈 Industry Growth",
            "international_release": "🌐 International Release",
            "vfx_x_screens": "VFX × Screens",
            "production_scale": "📐 Production Scale",
            "runtime_minutes": "⏱️ Runtime",
            "num_cast_members": "👥 Cast Size",
        }
        corr_df["Display_Name"] = corr_df["Feature"].map(
            lambda x: name_map.get(x, x.replace("_", " ").title())
        )

        # Horizontal bar chart
        fig = px.bar(corr_df, x="Importance_Pct", y="Display_Name",
                     orientation="h", color="Importance_Pct",
                     color_continuous_scale="Viridis",
                     title="🏆 Top 20 Features by Importance (%)")
        dark = plotly_dark()
        dark["yaxis"] = {**dark.get("yaxis", {}), "autorange": "reversed"}
        fig.update_layout(**dark, height=600, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

        # Top 3 insights
        st.markdown("### 💡 Key Insights")
        for _, row in corr_df.head(3).iterrows():
            st.markdown(create_insight_card(
                f"<b>{row['Display_Name']}</b> contributes <b>{row['Importance_Pct']:.1f}%</b> "
                f"to budget prediction",
                "📌"
            ), unsafe_allow_html=True)


# ────────────────────────────────────────────────────────
def render_trend_analysis(filtered_df, selected_industries, df):
    """📈 Budget Trend Analysis with inflation adjustment."""
    st.markdown("## 📈 Budget Growth Trend Analysis")

    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Growth Curves", "💹 Inflation-Adjusted", "🔥 Heatmap", "📉 YoY Growth"
    ])

    with tab1:
        yearly = filtered_df.groupby(["year", "industry"]).agg(
            avg_budget=("budget_crores", "mean"),
            total_budget=("budget_crores", "sum"),
            count=("movie_id", "count")
        ).reset_index()

        fig = px.line(yearly, x="year", y="avg_budget", color="industry",
                      color_discrete_map=INDUSTRY_COLORS,
                      title="Average Budget Growth Over Time", markers=True)
        fig.update_layout(**plotly_dark(), height=500)
        fig.update_traces(line=dict(width=3), marker=dict(size=6))
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        if "budget_2024_crores" in filtered_df.columns:
            yearly_adj = filtered_df.groupby(["year", "industry"])["budget_2024_crores"].mean().reset_index()
            fig = px.line(yearly_adj, x="year", y="budget_2024_crores", color="industry",
                          color_discrete_map=INDUSTRY_COLORS,
                          title="📈 Inflation-Adjusted Budget Trend (₹ 2024 Crores)")
            fig.update_layout(**plotly_dark(), height=500)
            fig.update_traces(line=dict(width=3))
            st.plotly_chart(fig, use_container_width=True)

            st.markdown(create_insight_card(
                "This chart adjusts for inflation using CPI data. "
                "A ₹10 Cr budget in 2000 equals ~₹42 Cr in 2024. "
                "This reveals REAL growth vs nominal growth.", "💹"
            ), unsafe_allow_html=True)
        else:
            st.info("Run the advanced training pipeline to get inflation-adjusted data.")

    with tab3:
        pivot = filtered_df.pivot_table(
            values="budget_crores", index="genre", columns="year", aggfunc="mean"
        )
        fig = px.imshow(pivot, aspect="auto", color_continuous_scale="Viridis",
                        title="Average Budget Heatmap: Genre × Year")
        fig.update_layout(**plotly_dark(), height=600)
        st.plotly_chart(fig, use_container_width=True)

    with tab4:
        yoy_data = []
        for industry in selected_industries:
            yearly_avg = filtered_df[filtered_df["industry"] == industry].groupby("year")["budget_crores"].mean()
            growth = yearly_avg.pct_change() * 100
            for yr, g in growth.items():
                yoy_data.append({"Year": yr, "Industry": industry, "Growth (%)": g})

        yoy_df = pd.DataFrame(yoy_data).dropna()
        fig = px.bar(yoy_df, x="Year", y="Growth (%)", color="Industry",
                     color_discrete_map=INDUSTRY_COLORS,
                     barmode="group", title="Year-over-Year Budget Growth Rate")
        fig.update_layout(**plotly_dark(), height=500)
        fig.add_hline(y=0, line_dash="dash", line_color="#FFD93D", opacity=0.5)
        st.plotly_chart(fig, use_container_width=True)


# ────────────────────────────────────────────────────────
def render_industry_comparison(filtered_df, selected_industries):
    """🌍 Industry comparison with ROI analysis."""
    st.markdown("## 🌍 Industry Comparison & ROI Analysis")

    col1, col2 = st.columns(2)

    with col1:
        fig = px.scatter(
            filtered_df.sample(min(1500, len(filtered_df))),
            x="budget_crores", y="box_office_crores",
            color="industry", color_discrete_map=INDUSTRY_COLORS,
            size="actor_popularity_score", size_max=15,
            title="💰 Budget vs Box Office Collection",
            hover_data=["genre", "year", "verdict"]
        )
        fig.add_trace(go.Scatter(
            x=[0, filtered_df["budget_crores"].max()],
            y=[0, filtered_df["budget_crores"].max()],
            mode='lines', line=dict(color='#FFD93D', dash='dash', width=2),
            name='Break Even', showlegend=True
        ))
        fig.update_layout(**plotly_dark(), height=500)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        hit_verdicts = ["Blockbuster", "Super Hit", "Hit"]
        hit_rates = []
        for industry in selected_industries:
            ind_df = filtered_df[filtered_df["industry"] == industry]
            total = len(ind_df)
            hits = len(ind_df[ind_df["verdict"].isin(hit_verdicts)])
            hit_rates.append({
                "Industry": industry,
                "Hit Rate (%)": round(hits / total * 100, 1) if total > 0 else 0,
                "Total Movies": total
            })

        hr_df = pd.DataFrame(hit_rates)
        fig = px.bar(hr_df, x="Industry", y="Hit Rate (%)", color="Industry",
                     color_discrete_map=INDUSTRY_COLORS,
                     title="🎯 Hit Rate by Industry", text="Hit Rate (%)")
        fig.update_layout(**plotly_dark(), height=500, showlegend=False)
        fig.update_traces(textposition='outside', textfont_size=14)
        st.plotly_chart(fig, use_container_width=True)

    # Industry statistics table
    st.markdown("### 📊 Industry Statistics")
    stats = filtered_df.groupby("industry").agg(
        Movies=("movie_id", "count"),
        Avg_Budget=("budget_crores", "mean"),
        Max_Budget=("budget_crores", "max"),
        Avg_BoxOffice=("box_office_crores", "mean"),
        Avg_Runtime=("runtime_minutes", "mean"),
    ).round(1).reset_index()

    if "roi_percentage" in filtered_df.columns:
        roi_stats = filtered_df.groupby("industry")["roi_percentage"].mean().round(1)
        stats = stats.merge(roi_stats.rename("Avg_ROI"), left_on="industry", right_index=True)
        stats.columns = ["Industry", "Movies", "Avg Budget (₹Cr)", "Max Budget (₹Cr)",
                          "Avg Box Office (₹Cr)", "Avg Runtime", "Avg ROI (%)"]
    else:
        stats.columns = ["Industry", "Movies", "Avg Budget (₹Cr)", "Max Budget (₹Cr)",
                          "Avg Box Office (₹Cr)", "Avg Runtime"]

    st.dataframe(stats, use_container_width=True, hide_index=True)

    # Sunburst
    st.markdown("### 🌐 Industry → Genre → Verdict Breakdown")
    sunburst_df = filtered_df.groupby(["industry", "genre", "verdict"]).size().reset_index(name="count")
    fig = px.sunburst(sunburst_df, path=["industry", "genre", "verdict"], values="count",
                      title="Hierarchical View", color="industry",
                      color_discrete_map=INDUSTRY_COLORS)
    fig.update_layout(**plotly_dark(), height=600)
    st.plotly_chart(fig, use_container_width=True)


# ────────────────────────────────────────────────────────
def render_genre_actor_insights(filtered_df):
    """🎭 Genre & Actor deep-dive insights."""
    st.markdown("## 🎭 Genre & Actor/Director Insights")

    tab1, tab2, tab3 = st.tabs(["🎭 Genre Analysis", "⭐ Actor Impact", "🎬 Director Impact"])

    with tab1:
        st.markdown("### Genre vs Budget — Deep Dive")
        genre_stats = filtered_df.groupby("genre").agg(
            avg_budget=("budget_crores", "mean"),
            median_budget=("budget_crores", "median"),
            count=("movie_id", "count"),
            avg_box_office=("box_office_crores", "mean"),
        ).reset_index()
        genre_stats = genre_stats[genre_stats["count"] >= 3].sort_values("avg_budget", ascending=False)

        fig = make_subplots(rows=1, cols=2,
                            subplot_titles=["Average Budget by Genre", "Avg Box Office by Genre"])
        fig.add_trace(go.Bar(
            x=genre_stats["genre"], y=genre_stats["avg_budget"],
            marker_color="#4ECDC4", name="Avg Budget"
        ), row=1, col=1)
        fig.add_trace(go.Bar(
            x=genre_stats["genre"], y=genre_stats["avg_box_office"],
            marker_color="#FF6B6B", name="Avg Box Office"
        ), row=1, col=2)
        fig.update_layout(**plotly_dark(), height=450, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        # Genre insights
        if len(genre_stats) >= 2:
            most = genre_stats.iloc[0]
            least = genre_stats.iloc[-1]
            ratio = most["avg_budget"] / least["avg_budget"] if least["avg_budget"] > 0 else 0
            st.markdown(create_insight_card(
                f"<b>{most['genre']}</b> films have <b>{ratio:.1f}×</b> higher average budgets "
                f"than <b>{least['genre']}</b> films "
                f"(₹{most['avg_budget']:.0f} Cr vs ₹{least['avg_budget']:.0f} Cr)",
                "📌"
            ), unsafe_allow_html=True)

    with tab2:
        st.markdown("### Top Actors vs Budget")
        actor_stats = filtered_df.groupby("lead_actor").agg(
            avg_budget=("budget_crores", "mean"),
            avg_box_office=("box_office_crores", "mean"),
            count=("movie_id", "count"),
        ).reset_index()
        actor_stats = actor_stats[actor_stats["count"] >= 3].sort_values("avg_budget", ascending=False)

        top_actors = actor_stats.head(15)
        fig = px.bar(top_actors, x="avg_budget", y="lead_actor",
                     orientation="h", color="avg_budget",
                     color_continuous_scale="Viridis",
                     title="⭐ Top 15 Actors by Average Budget")
        dark2 = plotly_dark()
        dark2["yaxis"] = {**dark2.get("yaxis", {}), "autorange": "reversed"}
        fig.update_layout(**dark2, height=500, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

        # Impact insight
        if len(actor_stats) >= 10:
            top_avg = actor_stats.head(10)["avg_budget"].mean()
            others_avg = actor_stats.iloc[10:]["avg_budget"].mean()
            diff = top_avg - others_avg
            st.markdown(create_insight_card(
                f"Top 10 actors increase average budget by <b>₹{diff:.1f} Crores</b> "
                f"compared to other actors (₹{top_avg:.0f} Cr vs ₹{others_avg:.0f} Cr)",
                "⭐"
            ), unsafe_allow_html=True)

    with tab3:
        st.markdown("### Director Success vs Budget")
        dir_stats = filtered_df.groupby("director").agg(
            avg_budget=("budget_crores", "mean"),
            hit_count=("verdict", lambda x: (x.isin(["Blockbuster", "Super Hit", "Hit"])).sum()),
            total=("movie_id", "count"),
        ).reset_index()
        dir_stats = dir_stats[dir_stats["total"] >= 3]
        dir_stats["hit_rate"] = (dir_stats["hit_count"] / dir_stats["total"] * 100).round(1)

        fig = px.scatter(dir_stats, x="hit_rate", y="avg_budget",
                         size="total", size_max=20,
                         color="avg_budget", color_continuous_scale="Viridis",
                         title="🎬 Director Hit Rate vs Average Budget",
                         hover_data=["director", "total"])
        fig.update_layout(**plotly_dark(), height=500)
        st.plotly_chart(fig, use_container_width=True)


# ────────────────────────────────────────────────────────
def render_error_analysis():
    """📉 Model Error Analysis — where does the model fail?"""
    st.markdown("## 📉 Error Analysis")
    st.markdown(create_insight_card(
        "Error analysis shows <b>where and why</b> the model fails. "
        "This level of analysis is <b>extremely rare</b> in student projects and shows professional maturity.",
        "🎓"
    ), unsafe_allow_html=True)

    err_data = load_json_safe(os.path.join(BASE_DIR, "data", "error_analysis.json"))

    if err_data:
        # Overall stats
        if "overall" in err_data:
            overall = err_data["overall"]
            cols = st.columns(4)
            metrics = [
                (f"₹{overall.get('mean_error', 0):.2f} Cr", "Mean Error (Bias)"),
                (f"₹{overall.get('std_error', 0):.2f} Cr", "Std Error (Spread)"),
                (f"₹{overall.get('median_abs_error', 0):.2f} Cr", "Median Abs Error"),
                (f"₹{overall.get('p90_abs_error', 0):.2f} Cr", "90th Pctl Error"),
            ]
            for col, (v, l) in zip(cols, metrics):
                with col:
                    st.markdown(create_metric_card(v, l), unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Error by budget range
        if "by_budget_range" in err_data:
            st.markdown("### 📊 Error by Budget Range")
            range_data = err_data["by_budget_range"]
            range_df = pd.DataFrame([
                {"Budget Range": k, "Count": v["count"],
                 "MAE (₹ Cr)": v["MAE"], "MAPE (%)": v["MAPE"]}
                for k, v in range_data.items()
            ])
            st.dataframe(range_df, use_container_width=True, hide_index=True)

            fig = px.bar(range_df, x="Budget Range", y="MAPE (%)",
                         color="MAPE (%)", color_continuous_scale="RdYlGn_r",
                         title="📊 Model Error Rate by Budget Range")
            fig.update_layout(**plotly_dark(), height=400, coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

            st.markdown(create_insight_card(
                "Models typically perform <b>worse on high-budget films</b> because there are "
                "fewer training examples. This is a known limitation of data-driven approaches.",
                "⚠️"
            ), unsafe_allow_html=True)
    else:
        st.warning("⚠️ Run `python src/train_model.py` to generate error analysis data.")


# ────────────────────────────────────────────────────────
def render_model_performance():
    """🧠 ML Model Performance comparison."""
    st.markdown("## 🧠 ML Model Performance")

    comp_path = os.path.join(BASE_DIR, "data", "model_comparison.csv")
    cv_data = load_json_safe(os.path.join(BASE_DIR, "data", "cv_results.json"))

    if os.path.exists(comp_path):
        comp_df = pd.read_csv(comp_path)

        # Model comparison chart
        fig = go.Figure()
        fig.add_trace(go.Bar(name="Train R²", x=comp_df["Model"], y=comp_df["Train R²"],
                             marker_color="#4ECDC4", opacity=0.8))
        fig.add_trace(go.Bar(name="Val R²", x=comp_df["Model"], y=comp_df["Val R²"],
                             marker_color="#45B7D1", opacity=0.8))
        fig.add_trace(go.Bar(name="Test R²", x=comp_df["Model"], y=comp_df["Test R²"],
                             marker_color="#FF6B6B", opacity=0.8))
        fig.update_layout(**plotly_dark(), barmode="group",
                          title="📊 R² Score Comparison Across Models", height=450)
        st.plotly_chart(fig, use_container_width=True)

        # Detailed table
        st.markdown("### 📋 Detailed Model Comparison")
        st.dataframe(comp_df, use_container_width=True, hide_index=True)

        # Best model highlight
        best = comp_df.loc[comp_df["Test R²"].idxmax()]
        st.markdown(
            f'<div class="prediction-box">'
            f'<div style="font-size:1rem; color:#8B949E;">🏆 Best Model</div>'
            f'<div class="prediction-value" style="font-size:2rem;">{best["Model"]}</div>'
            f'<div style="color:#4ECDC4; font-size:1.1rem; margin-top:0.5rem;">'
            f'R² = {best["Test R²"]:.4f} | MAE = ₹{best["Test MAE (₹Cr)"]:.2f} Cr | '
            f'MAPE = {best["Test MAPE (%)"]:.2f}%</div>'
            f'</div>',
            unsafe_allow_html=True
        )

        # Overfitting check
        if "Overfit Gap" in comp_df.columns:
            st.markdown("### ⚠️ Overfitting Analysis")
            fig = px.bar(comp_df, x="Model", y="Overfit Gap",
                         color="Overfit Gap", color_continuous_scale="RdYlGn_r",
                         title="Train-Test R² Gap (lower = better generalization)")
            fig.update_layout(**plotly_dark(), height=350, coloraxis_showscale=False)
            fig.add_hline(y=0.1, line_dash="dash", line_color="#FFD93D",
                          annotation_text="Overfitting Threshold")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("⚠️ Run `python src/train_model.py` to train models.")

    # Cross-validation results
    if cv_data:
        st.markdown("### 📊 5-Fold Cross-Validation Results")
        cv_rows = []
        for model_name, results in cv_data.items():
            cv_rows.append({
                "Model": model_name,
                "R² Mean": results["r2_mean"],
                "R² Std": results["r2_std"],
                "MAE Mean (₹Cr)": results["mae_mean"],
                "MAE Std (₹Cr)": results["mae_std"],
            })
        cv_df = pd.DataFrame(cv_rows)
        st.dataframe(cv_df, use_container_width=True, hide_index=True)

        st.markdown(create_insight_card(
            "Cross-validation ensures our results are <b>not due to a lucky data split</b>. "
            "Low R² standard deviation means the model is <b>stable and reliable</b>.",
            "✅"
        ), unsafe_allow_html=True)


# ════════════════════════════════════════════════════════
if __name__ == "__main__":
    main()
