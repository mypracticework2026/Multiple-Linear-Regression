import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Dubai Property Price Predictor",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;900&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Background */
    .stApp {
        background: linear-gradient(135deg, #0a0e1a 0%, #0d1527 50%, #0a1020 100%);
        color: #e8eaf6;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1b3e 0%, #091428 100%);
        border-right: 1px solid #1e3a5f;
    }
    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: #00d4ff;
    }

    /* Hero banner */
    .hero-banner {
        background: linear-gradient(135deg, #1a1f3e 0%, #0d2b5e 50%, #1a1f3e 100%);
        border: 1px solid #2a4a7f;
        border-radius: 16px;
        padding: 36px 40px;
        margin-bottom: 28px;
        position: relative;
        overflow: hidden;
    }
    .hero-banner::before {
        content: '';
        position: absolute;
        top: -40px; right: -40px;
        width: 200px; height: 200px;
        background: radial-gradient(circle, rgba(0,212,255,0.15) 0%, transparent 70%);
        border-radius: 50%;
    }
    .hero-title {
        font-size: 2.6rem;
        font-weight: 900;
        background: linear-gradient(90deg, #00d4ff, #a855f7, #f59e0b);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        line-height: 1.2;
        margin: 0 0 10px 0;
    }
    .hero-sub {
        color: #94a3b8;
        font-size: 1.05rem;
        font-weight: 300;
        margin: 0;
    }

    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #0f1f3d 0%, #1a2f50 100%);
        border: 1px solid #2a4a7f;
        border-radius: 14px;
        padding: 22px 24px;
        text-align: center;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 30px rgba(0,212,255,0.15);
    }
    .metric-label {
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #64748b;
        margin-bottom: 8px;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 800;
        color: #00d4ff;
    }
    .metric-value.gold   { color: #f59e0b; }
    .metric-value.purple { color: #a855f7; }
    .metric-value.green  { color: #10b981; }

    /* Prediction result */
    .prediction-box {
        background: linear-gradient(135deg, #0d2b1a 0%, #0a3d20 100%);
        border: 2px solid #10b981;
        border-radius: 16px;
        padding: 32px;
        text-align: center;
        box-shadow: 0 0 40px rgba(16,185,129,0.2);
        margin-top: 10px;
    }
    .prediction-label {
        font-size: 0.85rem;
        font-weight: 600;
        letter-spacing: 0.15em;
        text-transform: uppercase;
        color: #10b981;
        margin-bottom: 12px;
    }
    .prediction-price {
        font-size: 3.2rem;
        font-weight: 900;
        color: #ffffff;
        line-height: 1;
    }
    .prediction-sub {
        font-size: 0.9rem;
        color: #6ee7b7;
        margin-top: 8px;
        font-weight: 300;
    }

    /* Section headers */
    .section-header {
        font-size: 1.1rem;
        font-weight: 700;
        color: #00d4ff;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        border-left: 4px solid #00d4ff;
        padding-left: 14px;
        margin: 28px 0 18px 0;
    }

    /* Input container */
    .input-section {
        background: linear-gradient(135deg, #0f1f3d 0%, #111827 100%);
        border: 1px solid #1e3a5f;
        border-radius: 14px;
        padding: 28px 28px 12px 28px;
        margin-bottom: 20px;
    }

    /* Select boxes and sliders */
    .stSelectbox > div > div {
        background-color: #0d1b3e !important;
        border: 1px solid #2a4a7f !important;
        border-radius: 10px !important;
        color: #e2e8f0 !important;
    }
    .stSlider > div > div > div {
        background: linear-gradient(90deg, #00d4ff, #a855f7) !important;
    }
    .stNumberInput > div > div > input {
        background-color: #0d1b3e !important;
        border: 1px solid #2a4a7f !important;
        color: #e2e8f0 !important;
        border-radius: 10px !important;
    }

    /* Button */
    .stButton > button {
        background: linear-gradient(135deg, #00d4ff 0%, #a855f7 100%);
        color: #000000;
        font-weight: 800;
        font-size: 1rem;
        letter-spacing: 0.05em;
        border: none;
        border-radius: 12px;
        padding: 14px 0;
        width: 100%;
        transition: opacity 0.2s, transform 0.2s;
        box-shadow: 0 4px 20px rgba(0,212,255,0.3);
    }
    .stButton > button:hover {
        opacity: 0.9;
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(0,212,255,0.5);
    }

    /* Feature badge pills */
    .badge-row {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        margin-bottom: 20px;
    }
    .badge {
        padding: 6px 14px;
        border-radius: 99px;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.05em;
    }
    .badge-blue   { background: rgba(0,212,255,0.15); color: #00d4ff; border: 1px solid rgba(0,212,255,0.3); }
    .badge-purple { background: rgba(168,85,247,0.15); color: #a855f7; border: 1px solid rgba(168,85,247,0.3); }
    .badge-gold   { background: rgba(245,158,11,0.15); color: #f59e0b; border: 1px solid rgba(245,158,11,0.3); }
    .badge-green  { background: rgba(16,185,129,0.15); color: #10b981; border: 1px solid rgba(16,185,129,0.3); }
    .badge-pink   { background: rgba(236,72,153,0.15); color: #ec4899; border: 1px solid rgba(236,72,153,0.3); }

    /* Chart bg */
    .chart-container {
        background: #0f1f3d;
        border: 1px solid #1e3a5f;
        border-radius: 14px;
        padding: 20px;
    }

    /* Divider */
    hr { border-color: #1e3a5f; }

    /* Hide Streamlit branding */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ─── Load Data & Model ────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_data
def load_data():
    return pd.read_csv(os.path.join(BASE_DIR, "Dubai-property-prices2.csv"))

@st.cache_resource
def load_model():
    with open(os.path.join(BASE_DIR, "MultiRegmodel.pkl"), "rb") as f:
        return pickle.load(f)

df = load_data()
model = load_model()

# Rebuild feature names (same as training)
df_encoded = pd.get_dummies(df, drop_first=True)
feature_names = df_encoded.drop("Price_AED", axis=1).columns.tolist()


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 10px 0 20px 0;'>
        <div style='font-size:2.5rem;'>🏙️</div>
        <div style='font-size:1.1rem; font-weight:800; color:#00d4ff; letter-spacing:0.05em;'>DUBAI PROPERTY</div>
        <div style='font-size:0.75rem; color:#64748b; letter-spacing:0.15em; text-transform:uppercase;'>Price Intelligence</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🎛️ Configure Property")

    # LOCATION
    st.markdown("**📍 Location**")
    location = st.selectbox("", sorted(df["Location"].unique()), label_visibility="collapsed", key="loc")

    # PROPERTY TYPE
    st.markdown("**🏢 Property Type**")
    property_type = st.selectbox("", sorted(df["Property_Type"].unique()), label_visibility="collapsed", key="ptype")

    # BEDROOMS
    st.markdown("**🛏️ Bedrooms**")
    bedrooms = st.selectbox("", sorted(df["Bedrooms"].unique()), label_visibility="collapsed", key="beds")

    # BATHROOMS
    st.markdown("**🚿 Bathrooms**")
    bathrooms = st.selectbox("", sorted(df["Bathrooms"].unique()), label_visibility="collapsed", key="baths")

    # VIEW
    st.markdown("**🌅 View**")
    view_options = sorted(df["View"].unique())
    view = st.selectbox("", view_options, label_visibility="collapsed", key="view")

    # FURNISHED
    st.markdown("**🛋️ Furnished**")
    furnished = st.selectbox("", ["Yes", "No"], label_visibility="collapsed", key="furn")

    # SIZE
    st.markdown("**📐 Size (sq ft)**")
    min_size = int(df["Size_sqft"].min())
    max_size = int(df["Size_sqft"].max())
    size = st.slider("", min_size, max_size, int(df["Size_sqft"].median()), step=50,
                     label_visibility="collapsed", key="size")

    # AGE
    st.markdown("**🏗️ Property Age (years)**")
    age = st.slider("", int(df["Age"].min()), int(df["Age"].max()),
                    int(df["Age"].median()), label_visibility="collapsed", key="age")

    st.markdown("---")

    predict_btn = st.button("🔮  PREDICT PRICE", use_container_width=True)


# ─── Helper: prepare input ────────────────────────────────────────────────────
def prepare_input(location, property_type, bedrooms, bathrooms, size, age, view, furnished, feature_names):
    row = pd.DataFrame([{
        "Location": location,
        "Property_Type": property_type,
        "Bedrooms": bedrooms,
        "Bathrooms": bathrooms,
        "Size_sqft": size,
        "Age": age,
        "View": view,
        "Furnished": furnished
    }])
    row_encoded = pd.get_dummies(row)
    row_encoded = row_encoded.reindex(columns=feature_names, fill_value=0)
    return row_encoded


# ─── Main Layout ──────────────────────────────────────────────────────────────

# Hero
st.markdown("""
<div class="hero-banner">
    <div class="hero-title">Dubai Property Price Predictor</div>
    <p class="hero-sub">Machine learning–powered valuation across Dubai's most sought-after neighbourhoods</p>
</div>
""", unsafe_allow_html=True)

# Dataset KPI row
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Total Listings</div>
        <div class="metric-value">{len(df):,}</div>
    </div>""", unsafe_allow_html=True)
with col2:
    avg_price = int(df["Price_AED"].mean())
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Avg Price (AED)</div>
        <div class="metric-value gold">{avg_price:,}</div>
    </div>""", unsafe_allow_html=True)
with col3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Locations</div>
        <div class="metric-value purple">{df['Location'].nunique()}</div>
    </div>""", unsafe_allow_html=True)
with col4:
    avg_size = int(df["Size_sqft"].mean())
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Avg Size (sqft)</div>
        <div class="metric-value green">{avg_size:,}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Selected property badges
st.markdown(f"""
<div class="badge-row">
    <span class="badge badge-blue">📍 {location}</span>
    <span class="badge badge-purple">🏢 {property_type}</span>
    <span class="badge badge-gold">🛏️ {bedrooms} Bed</span>
    <span class="badge badge-green">🚿 {bathrooms} Bath</span>
    <span class="badge badge-pink">📐 {size:,} sqft</span>
    <span class="badge badge-blue">🌅 {view}</span>
    <span class="badge badge-purple">🛋️ {'Furnished' if furnished == 'Yes' else 'Unfurnished'}</span>
    <span class="badge badge-gold">🏗️ {age} yrs old</span>
</div>
""", unsafe_allow_html=True)

# ─── Prediction Result ────────────────────────────────────────────────────────
if predict_btn:
    input_data = prepare_input(location, property_type, bedrooms, bathrooms, size, age, view, furnished, feature_names)
    prediction = model.predict(input_data)[0]

    st.markdown(f"""
    <div class="prediction-box">
        <div class="prediction-label">✨ Estimated Market Value</div>
        <div class="prediction-price">AED {int(prediction):,}</div>
        <div class="prediction-sub">Based on {len(df)} Dubai property listings · Multiple Linear Regression</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Price per sqft
    price_per_sqft = int(prediction / size)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Price / sqft</div>
            <div class="metric-value gold">AED {price_per_sqft:,}</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        similar = df[(df["Location"] == location) & (df["Property_Type"] == property_type)]
        if len(similar) > 0:
            market_avg = int(similar["Price_AED"].mean())
            delta = int(prediction - market_avg)
            delta_pct = round((delta / market_avg) * 100, 1)
            color = "green" if delta <= 0 else "gold"
            label = "below" if delta <= 0 else "above"
            st.markdown(f"""<div class="metric-card">
                <div class="metric-label">vs Area Avg</div>
                <div class="metric-value {color}">{delta_pct:+}%</div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-label">vs Area Avg</div>
                <div class="metric-value">N/A</div>
            </div>""", unsafe_allow_html=True)
    with c3:
        budget_range = f"AED {int(prediction*0.95):,} – {int(prediction*1.05):,}"
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">±5% Range</div>
            <div class="metric-value purple" style="font-size:1rem; line-height:1.5;">{budget_range}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("---")

# ─── Charts Row ───────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">📊 Market Insights</div>', unsafe_allow_html=True)

chart_col1, chart_col2 = st.columns(2)

DARK_BG   = "#0f1f3d"
GRID_COL  = "#1e3a5f"
TEXT_COL  = "#94a3b8"
PALETTE   = ["#00d4ff", "#a855f7", "#f59e0b", "#10b981", "#ec4899",
             "#f97316", "#3b82f6", "#84cc16", "#e11d48", "#06b6d4"]

# Chart 1: Avg Price by Location
with chart_col1:
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(DARK_BG)

    loc_avg = df.groupby("Location")["Price_AED"].mean().sort_values(ascending=True)
    colors_loc = plt.cm.cool(np.linspace(0.2, 0.9, len(loc_avg)))
    bars = ax.barh(loc_avg.index, loc_avg.values / 1e6, color=colors_loc, edgecolor="none", height=0.6)

    ax.set_xlabel("Avg Price (AED Millions)", color=TEXT_COL, fontsize=9)
    ax.set_title("Average Price by Location", color="#00d4ff", fontsize=11, fontweight="bold", pad=12)
    ax.tick_params(colors=TEXT_COL, labelsize=8)
    ax.spines[["top","right","bottom","left"]].set_color(GRID_COL)
    ax.xaxis.grid(True, color=GRID_COL, linewidth=0.5)
    ax.set_axisbelow(True)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    st.markdown('</div>', unsafe_allow_html=True)

# Chart 2: Price by Property Type
with chart_col2:
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    fig2.patch.set_facecolor(DARK_BG)
    ax2.set_facecolor(DARK_BG)

    type_avg = df.groupby("Property_Type")["Price_AED"].mean().sort_values(ascending=False)
    ax2.bar(type_avg.index, type_avg.values / 1e6,
            color=["#00d4ff", "#a855f7"], edgecolor="none", width=0.5)

    ax2.set_ylabel("Avg Price (AED Millions)", color=TEXT_COL, fontsize=9)
    ax2.set_title("Price by Property Type", color="#a855f7", fontsize=11, fontweight="bold", pad=12)
    ax2.tick_params(colors=TEXT_COL, labelsize=9)
    ax2.spines[["top","right","bottom","left"]].set_color(GRID_COL)
    ax2.yaxis.grid(True, color=GRID_COL, linewidth=0.5)
    ax2.set_axisbelow(True)
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
chart_col3, chart_col4 = st.columns(2)

# Chart 3: Size vs Price scatter
with chart_col3:
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    fig3, ax3 = plt.subplots(figsize=(6, 4))
    fig3.patch.set_facecolor(DARK_BG)
    ax3.set_facecolor(DARK_BG)

    sc = ax3.scatter(df["Size_sqft"], df["Price_AED"] / 1e6,
                     c=df["Bedrooms"], cmap="plasma", alpha=0.7, s=45, edgecolors="none")
    cbar = plt.colorbar(sc, ax=ax3)
    cbar.set_label("Bedrooms", color=TEXT_COL, fontsize=8)
    cbar.ax.yaxis.set_tick_params(color=TEXT_COL)
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color=TEXT_COL, fontsize=7)

    ax3.set_xlabel("Size (sqft)", color=TEXT_COL, fontsize=9)
    ax3.set_ylabel("Price (AED Millions)", color=TEXT_COL, fontsize=9)
    ax3.set_title("Size vs Price (coloured by Bedrooms)", color="#f59e0b", fontsize=11, fontweight="bold", pad=12)
    ax3.tick_params(colors=TEXT_COL, labelsize=8)
    ax3.spines[["top","right","bottom","left"]].set_color(GRID_COL)
    ax3.xaxis.grid(True, color=GRID_COL, linewidth=0.5)
    ax3.yaxis.grid(True, color=GRID_COL, linewidth=0.5)
    ax3.set_axisbelow(True)
    plt.tight_layout()
    st.pyplot(fig3)
    plt.close()
    st.markdown('</div>', unsafe_allow_html=True)

# Chart 4: Furnished vs Unfurnished avg price
with chart_col4:
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    fig4, ax4 = plt.subplots(figsize=(6, 4))
    fig4.patch.set_facecolor(DARK_BG)
    ax4.set_facecolor(DARK_BG)

    view_avg = df.groupby("View")["Price_AED"].mean().sort_values(ascending=False)
    bar_colors = PALETTE[:len(view_avg)]
    ax4.bar(view_avg.index, view_avg.values / 1e6, color=bar_colors, edgecolor="none", width=0.5)

    ax4.set_ylabel("Avg Price (AED Millions)", color=TEXT_COL, fontsize=9)
    ax4.set_title("Average Price by View Type", color="#10b981", fontsize=11, fontweight="bold", pad=12)
    ax4.tick_params(colors=TEXT_COL, labelsize=8)
    ax4.spines[["top","right","bottom","left"]].set_color(GRID_COL)
    ax4.yaxis.grid(True, color=GRID_COL, linewidth=0.5)
    ax4.set_axisbelow(True)
    plt.tight_layout()
    st.pyplot(fig4)
    plt.close()
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")

# ─── Data Table ───────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">📋 Dataset Explorer</div>', unsafe_allow_html=True)

filter_loc = st.multiselect("Filter by Location", options=sorted(df["Location"].unique()),
                             default=[], placeholder="All locations")
filtered_df = df[df["Location"].isin(filter_loc)] if filter_loc else df

st.dataframe(
    filtered_df.style.background_gradient(subset=["Price_AED"], cmap="Blues")
               .format({"Price_AED": "AED {:,.0f}", "Size_sqft": "{:,.0f} sqft"}),
    use_container_width=True,
    height=320
)

st.markdown(f"""
<div style='text-align:center; color:#475569; font-size:0.8rem; padding: 20px 0 0 0;'>
    Built with ❤️ using Streamlit · Multiple Linear Regression · Dubai Property Dataset
</div>
""", unsafe_allow_html=True)
