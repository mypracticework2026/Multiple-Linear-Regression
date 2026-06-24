import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

st.set_page_config(
    page_title="Prestige Properties Dubai",
    page_icon="👑",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@300;400;600;700&family=Inter:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #080808;
    color: #e8e0d0;
}
.stApp {
    background: #080808;
}
section[data-testid="stSidebar"] {
    background: #0e0e0e;
    border-right: 1px solid #2a2218;
}
section[data-testid="stSidebar"] > div {
    padding-top: 0;
}
.stSelectbox label {
    color: #8a7a5a !important;
    font-size: 0.7rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.18em !important;
    text-transform: uppercase !important;
}
.stSelectbox > div > div {
    background-color: #111111 !important;
    border: 1px solid #2a2218 !important;
    border-radius: 4px !important;
    color: #e8e0d0 !important;
}
.stSelectbox > div > div:hover {
    border-color: #C9A84C !important;
}
.stSlider label {
    color: #8a7a5a !important;
    font-size: 0.7rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.18em !important;
    text-transform: uppercase !important;
}
.stSlider > div > div > div > div {
    background: #C9A84C !important;
}
.stButton > button {
    background: linear-gradient(135deg, #C9A84C 0%, #e8c96a 50%, #C9A84C 100%);
    color: #080808;
    font-weight: 700;
    font-size: 0.8rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    border: none;
    border-radius: 2px;
    padding: 16px 0;
    width: 100%;
    box-shadow: 0 4px 24px rgba(201,168,76,0.25);
}
.stButton > button:hover {
    box-shadow: 0 8px 40px rgba(201,168,76,0.5);
    transform: translateY(-1px);
}
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


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

df_encoded = pd.get_dummies(df, drop_first=True)
feature_names = df_encoded.drop("Price_AED", axis=1).columns.tolist()

def prepare_input(location, property_type, bedrooms, bathrooms, size, age, view, furnished):
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


with st.sidebar:
    st.markdown("""
    <div style="padding: 40px 24px 32px 24px; border-bottom: 1px solid #2a2218;">
        <div style="font-family: 'Cormorant Garamond', serif; font-size: 0.65rem; font-weight: 400;
                    letter-spacing: 0.35em; text-transform: uppercase; color: #C9A84C; margin-bottom: 6px;">
            Prestige Properties
        </div>
        <div style="font-family: 'Cormorant Garamond', serif; font-size: 1.9rem; font-weight: 300;
                    color: #f0e8d8; line-height: 1.1; letter-spacing: 0.05em;">
            Dubai
        </div>
        <div style="width: 32px; height: 1px; background: #C9A84C; margin-top: 14px;"></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='padding: 24px 8px 8px 8px;'>", unsafe_allow_html=True)

    location      = st.selectbox("Location",             sorted(df["Location"].unique()))
    property_type = st.selectbox("Property Type",        sorted(df["Property_Type"].unique()))
    bedrooms      = st.selectbox("Bedrooms",             sorted(df["Bedrooms"].unique()))
    bathrooms     = st.selectbox("Bathrooms",            sorted(df["Bathrooms"].unique()))
    view          = st.selectbox("View",                 sorted(df["View"].unique()))
    furnished     = st.selectbox("Furnished",            ["Yes", "No"])

    st.markdown("<br>", unsafe_allow_html=True)

    min_s, max_s = int(df["Size_sqft"].min()), int(df["Size_sqft"].max())
    size = st.slider("Size (sq ft)", min_s, max_s, int(df["Size_sqft"].median()), step=50)

    min_a, max_a = int(df["Age"].min()), int(df["Age"].max())
    age = st.slider("Property Age (years)", min_a, max_a, int(df["Age"].median()))

    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("OBTAIN VALUATION")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("""
    <div style="position: fixed; bottom: 0; left: 0; width: 280px; padding: 16px 24px;
                border-top: 1px solid #2a2218; background: #0e0e0e;">
        <div style="font-size: 0.65rem; color: #4a4030; letter-spacing: 0.1em; text-align: center;">
            POWERED BY MACHINE LEARNING
        </div>
    </div>
    """, unsafe_allow_html=True)


st.markdown("""
<div style="height: 2px; background: linear-gradient(90deg, transparent, #C9A84C, transparent);
            margin-bottom: 0;"></div>
""", unsafe_allow_html=True)

st.markdown("""
<div style="padding: 64px 48px 48px 48px; background: #080808;">
    <div style="font-family: 'Cormorant Garamond', serif; font-size: 0.7rem; font-weight: 400;
                letter-spacing: 0.4em; text-transform: uppercase; color: #C9A84C; margin-bottom: 20px;">
        Property Valuation - Dubai
    </div>
    <div style="font-family: 'Cormorant Garamond', serif; font-size: 3.8rem; font-weight: 300;
                color: #f0e8d8; line-height: 1.05; letter-spacing: 0.02em; margin-bottom: 20px;">
        Discover Your Property's True Value
    </div>
    <div style="font-family: 'Inter', sans-serif; font-size: 0.9rem; font-weight: 300;
                color: #6a5c40; max-width: 480px; line-height: 1.8; letter-spacing: 0.02em;">
        Configure your property on the left and receive an instant AI-powered
        market valuation, trusted, precise, and refined.
    </div>
    <div style="width: 48px; height: 1px; background: #C9A84C; margin-top: 32px;"></div>
</div>
""", unsafe_allow_html=True)


if predict_btn:
    input_data = prepare_input(location, property_type, bedrooms, bathrooms,
                               size, age, view, furnished)
    prediction = model.predict(input_data)[0]

    similar = df[
        (df["Location"] == location) &
        (df["Property_Type"] == property_type)
    ]
    has_comp = len(similar) >= 2
    delta_pct = 0.0
    if has_comp:
        market_avg = similar["Price_AED"].mean()
        delta_pct  = ((prediction - market_avg) / market_avg) * 100

    price_per_sqft = int(prediction / size)
    range_low      = int(prediction * 0.95)
    range_high     = int(prediction * 1.05)
    delta_str      = f"{delta_pct:+.1f}%"
    delta_color    = "#a8d4a0" if delta_pct <= 0 else "#C9A84C"

    st.markdown(f"""
    <div style="margin: 0 48px 48px 48px; padding: 56px 48px;
                background: linear-gradient(135deg, #0e0c08 0%, #141008 100%);
                border: 1px solid #2a2218;
                border-top: 2px solid #C9A84C;
                position: relative; overflow: hidden;">
        <div style="position:absolute; top:-60px; right:-60px; width:300px; height:300px;
                    background: radial-gradient(circle, rgba(201,168,76,0.07) 0%, transparent 70%);
                    border-radius:50%; pointer-events:none;"></div>
        <div style="font-family:'Cormorant Garamond', serif; font-size:0.65rem; font-weight:400;
                    letter-spacing:0.4em; text-transform:uppercase; color:#C9A84C; margin-bottom:20px;">
            Estimated Market Value
        </div>
        <div style="font-family:'Cormorant Garamond', serif; font-size:4.5rem; font-weight:300;
                    color:#f0e8d8; line-height:1; letter-spacing:0.02em; margin-bottom:8px;">
            AED {int(prediction):,}
        </div>
        <div style="font-size:0.8rem; color:#4a4030; letter-spacing:0.1em; margin-bottom:40px;">
            {location} &nbsp;·&nbsp; {property_type} &nbsp;·&nbsp; {size:,} sq ft
        </div>
        <div style="height:1px; background: linear-gradient(90deg, #2a2218, transparent);
                    margin-bottom:40px;"></div>
        <div style="display:flex; gap:48px; flex-wrap:wrap;">
            <div>
                <div style="font-size:0.62rem; letter-spacing:0.25em; text-transform:uppercase;
                            color:#4a4030; margin-bottom:8px;">Price per sq ft</div>
                <div style="font-family:'Cormorant Garamond',serif; font-size:1.8rem;
                            font-weight:400; color:#C9A84C;">
                    AED {price_per_sqft:,}
                </div>
            </div>
            <div style="width:1px; background:#2a2218;"></div>
            <div>
                <div style="font-size:0.62rem; letter-spacing:0.25em; text-transform:uppercase;
                            color:#4a4030; margin-bottom:8px;">Valuation Range</div>
                <div style="font-family:'Cormorant Garamond',serif; font-size:1.8rem;
                            font-weight:400; color:#e8e0d0;">
                    AED {range_low:,} to {range_high:,}
                </div>
            </div>
            {"" if not has_comp else f'<div style="width:1px; background:#2a2218;"></div><div><div style="font-size:0.62rem; letter-spacing:0.25em; text-transform:uppercase; color:#4a4030; margin-bottom:8px;">vs Area Average</div><div style="font-family:Cormorant Garamond,serif; font-size:1.8rem; font-weight:400; color:{delta_color};">{delta_str}</div></div>'}
        </div>
    </div>
    """, unsafe_allow_html=True)

    furnished_label = "Fully Furnished" if furnished == "Yes" else "Unfurnished"

    st.markdown(f"""
    <div style="margin: 0 48px 48px 48px;">
        <div style="font-family:'Cormorant Garamond',serif; font-size:0.65rem; letter-spacing:0.35em;
                    text-transform:uppercase; color:#C9A84C; margin-bottom:24px;">
            Property Summary
        </div>
        <div style="display:grid; grid-template-columns: repeat(4, 1fr); gap:2px;">
            <div style="background:#0e0e0e; border:1px solid #1e1a12; padding:24px 20px;">
                <div style="font-size:0.6rem; letter-spacing:0.2em; text-transform:uppercase;
                            color:#4a4030; margin-bottom:10px;">Location</div>
                <div style="font-family:'Cormorant Garamond',serif; font-size:1.15rem;
                            color:#f0e8d8;">{location}</div>
            </div>
            <div style="background:#0e0e0e; border:1px solid #1e1a12; padding:24px 20px;">
                <div style="font-size:0.6rem; letter-spacing:0.2em; text-transform:uppercase;
                            color:#4a4030; margin-bottom:10px;">Type</div>
                <div style="font-family:'Cormorant Garamond',serif; font-size:1.15rem;
                            color:#f0e8d8;">{property_type}</div>
            </div>
            <div style="background:#0e0e0e; border:1px solid #1e1a12; padding:24px 20px;">
                <div style="font-size:0.6rem; letter-spacing:0.2em; text-transform:uppercase;
                            color:#4a4030; margin-bottom:10px;">Configuration</div>
                <div style="font-family:'Cormorant Garamond',serif; font-size:1.15rem;
                            color:#f0e8d8;">{bedrooms} Bed &nbsp;·&nbsp; {bathrooms} Bath</div>
            </div>
            <div style="background:#0e0e0e; border:1px solid #1e1a12; padding:24px 20px;">
                <div style="font-size:0.6rem; letter-spacing:0.2em; text-transform:uppercase;
                            color:#4a4030; margin-bottom:10px;">View</div>
                <div style="font-family:'Cormorant Garamond',serif; font-size:1.15rem;
                            color:#f0e8d8;">{view}</div>
            </div>
            <div style="background:#0e0e0e; border:1px solid #1e1a12; padding:24px 20px;">
                <div style="font-size:0.6rem; letter-spacing:0.2em; text-transform:uppercase;
                            color:#4a4030; margin-bottom:10px;">Size</div>
                <div style="font-family:'Cormorant Garamond',serif; font-size:1.15rem;
                            color:#f0e8d8;">{size:,} sq ft</div>
            </div>
            <div style="background:#0e0e0e; border:1px solid #1e1a12; padding:24px 20px;">
                <div style="font-size:0.6rem; letter-spacing:0.2em; text-transform:uppercase;
                            color:#4a4030; margin-bottom:10px;">Age</div>
                <div style="font-family:'Cormorant Garamond',serif; font-size:1.15rem;
                            color:#f0e8d8;">{age} Years</div>
            </div>
            <div style="background:#0e0e0e; border:1px solid #1e1a12; padding:24px 20px;">
                <div style="font-size:0.6rem; letter-spacing:0.2em; text-transform:uppercase;
                            color:#4a4030; margin-bottom:10px;">Furnishing</div>
                <div style="font-family:'Cormorant Garamond',serif; font-size:1.15rem;
                            color:#f0e8d8;">{furnished_label}</div>
            </div>
            <div style="background: linear-gradient(135deg,#1a1408,#100e06);
                        border:1px solid #2a2218; padding:24px 20px;">
                <div style="font-size:0.6rem; letter-spacing:0.2em; text-transform:uppercase;
                            color:#C9A84C; margin-bottom:10px;">Model</div>
                <div style="font-family:'Cormorant Garamond',serif; font-size:1.15rem;
                            color:#C9A84C;">ML Regression</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if has_comp:
        top_similar = similar.nlargest(3, "Price_AED")
        cards_html = ""
        for i, (_, row) in enumerate(top_similar.iterrows()):
            is_highlight = i == 0
            bg         = "linear-gradient(135deg,#1a1408,#100e06)" if is_highlight else "#0e0e0e"
            bdr        = "#C9A84C" if is_highlight else "#1e1a12"
            pcol       = "#C9A84C" if is_highlight else "#f0e8d8"
            badge      = "<div style='position:absolute;top:16px;right:16px;font-size:0.55rem;letter-spacing:0.2em;color:#C9A84C;background:rgba(201,168,76,0.1);padding:4px 10px;border:1px solid rgba(201,168,76,0.3);'>HIGHEST</div>" if is_highlight else ""
            furn_label = "Furnished" if row["Furnished"] == "Yes" else "Unfurnished"
            cards_html += f"""
            <div style="background:{bg}; border:1px solid {bdr}; padding:28px 24px; position:relative;">
                {badge}
                <div style="font-family:'Cormorant Garamond',serif; font-size:1.6rem; font-weight:300;
                            color:{pcol}; margin-bottom:6px;">
                    AED {int(row['Price_AED']):,}
                </div>
                <div style="font-size:0.7rem; color:#4a4030; letter-spacing:0.05em; margin-bottom:16px;">
                    {int(row['Bedrooms'])} Bed - {int(row['Bathrooms'])} Bath - {int(row['Size_sqft']):,} sqft
                </div>
                <div style="font-size:0.65rem; color:#3a3020; letter-spacing:0.1em;">
                    {row['View']} View &nbsp;·&nbsp; {furn_label}
                </div>
            </div>
            """

        st.markdown(f"""
        <div style="margin: 0 48px 48px 48px;">
            <div style="font-family:'Cormorant Garamond',serif; font-size:0.65rem; letter-spacing:0.35em;
                        text-transform:uppercase; color:#C9A84C; margin-bottom:24px;">
                Comparable Properties in {location}
            </div>
            <div style="display:grid; grid-template-columns: repeat(3, 1fr); gap:2px;">
                {cards_html}
            </div>
        </div>
        """, unsafe_allow_html=True)

else:
    st.markdown("""
    <div style="margin: 0 48px; padding: 80px 48px; text-align:center;
                border: 1px solid #1a1610; background:#0a0906;">
        <div style="font-size:2rem; margin-bottom:20px; opacity:0.3;">◆</div>
        <div style="font-family:'Cormorant Garamond',serif; font-size:1.4rem; font-weight:300;
                    color:#4a4030; letter-spacing:0.05em; margin-bottom:12px;">
            Configure your property details
        </div>
        <div style="font-size:0.8rem; color:#2a2218; letter-spacing:0.1em;">
            SELECT OPTIONS ON THE LEFT - THEN OBTAIN VALUATION
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<div style="margin: 64px 48px 32px 48px; padding-top: 24px;
            border-top: 1px solid #1a1610;
            display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:12px;">
    <div style="font-family:'Cormorant Garamond',serif; font-size:0.8rem;
                color:#2a2218; letter-spacing:0.15em;">
        PRESTIGE PROPERTIES - DUBAI
    </div>
    <div style="font-size:0.65rem; color:#1e1a12; letter-spacing:0.1em;">
        AI-POWERED VALUATION - FOR ILLUSTRATIVE PURPOSES
    </div>
</div>
""", unsafe_allow_html=True)
