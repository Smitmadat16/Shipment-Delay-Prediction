import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.set_page_config(
    page_title="Shipment Delay Predictor",
    page_icon="🚚",
    layout="centered"
)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }
    .main {
        background-color: #0f1117;
    }
    .title-block {
        background: linear-gradient(135deg, #1a1f2e, #2d3561);
        border-radius: 16px;
        padding: 2rem 2.5rem;
        margin-bottom: 2rem;
        border: 1px solid #3d4a7a;
    }
    .title-block h1 {
        font-family: 'Space Mono', monospace;
        color: #ffffff;
        font-size: 1.8rem;
        margin: 0;
    }
    .title-block p {
        color: #8892b0;
        margin: 0.5rem 0 0 0;
        font-size: 0.95rem;
    }
    .result-delayed {
        background: linear-gradient(135deg, #2d1b1b, #4a1f1f);
        border: 1px solid #e53e3e;
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        margin-top: 1.5rem;
    }
    .result-ontime {
        background: linear-gradient(135deg, #1b2d1b, #1f4a2a);
        border: 1px solid #38a169;
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        margin-top: 1.5rem;
    }
    .result-emoji {
        font-size: 3.5rem;
        margin-bottom: 0.5rem;
    }
    .result-label {
        font-family: 'Space Mono', monospace;
        font-size: 1.6rem;
        font-weight: 700;
        margin: 0.5rem 0;
    }
    .result-prob {
        font-size: 1rem;
        color: #a0aec0;
        margin-top: 0.5rem;
    }
    .prob-bar-container {
        background: #1a1f2e;
        border-radius: 999px;
        height: 12px;
        margin: 1rem auto;
        max-width: 300px;
        overflow: hidden;
    }
    .stButton > button {
        background: linear-gradient(135deg, #3d4a7a, #5a67d8);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 2.5rem;
        font-family: 'Space Mono', monospace;
        font-size: 1rem;
        font-weight: 700;
        width: 100%;
        cursor: pointer;
        transition: all 0.2s;
        letter-spacing: 0.05em;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #5a67d8, #667eea);
        transform: translateY(-1px);
    }
    .info-card {
        background: #1a1f2e;
        border-radius: 12px;
        padding: 1rem 1.25rem;
        border: 1px solid #2d3561;
        margin-bottom: 1rem;
        font-size: 0.875rem;
        color: #8892b0;
    }
    .section-label {
        font-family: 'Space Mono', monospace;
        font-size: 0.75rem;
        color: #5a67d8;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        margin-bottom: 0.75rem;
        margin-top: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    with open('rf_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('model_columns.pkl', 'rb') as f:
        columns = pickle.load(f)
    return model, columns

try:
    model, model_columns = load_model()
    model_loaded = True
except FileNotFoundError:
    model_loaded = False

st.markdown("""
<div class="title-block">
    <h1>🚚 Shipment Delay Predictor</h1>
    <p>Data-Driven Analysis and Prediction of Shipment Delays · CS 210 Final Project</p>
    <p style="color:#5a67d8; font-size:0.8rem; margin-top:0.5rem;">Smit Madat · snm178 · Random Forest Model (AUC = 0.795)</p>
</div>
""", unsafe_allow_html=True)

if not model_loaded:
    st.error(" Model files not found! Please run `save_model.py` in your notebook first to generate `rf_model.pkl` and `model_columns.pkl`.")
    st.stop()
st.markdown('<div class="section-label">📦 Shipment Details</div>', unsafe_allow_html=True)
col1, col2 = st.columns(2)

with col1:
    shipping_mode = st.selectbox("Shipping Mode", [
        "Standard Class", "First Class", "Second Class", "Same Day"
    ])
    order_region = st.selectbox("Order Region", [
        "Western Europe", "Central America", "South America",
        "Southeast Asia", "South Asia", "North America",
        "Oceania", "East Africa", "West Africa", "Central Africa",
        "Northern Europe", "Eastern Europe", "Southern Europe",
        "West Asia", "Eastern Asia", "Caribbean"
    ])
    category_name = st.selectbox("Product Category", [
        "Sporting Goods", "Cleats", "Men's Footwear", "Women's Apparel",
        "Indoor/Outdoor Games", "Electronics", "Fitness Accessories",
        "Cameras", "Computers", "Fishing", "Golf Bags & Carts",
        "Lacrosse", "Tennis & Racquet", "Trade-In"
    ])
with col2:
    order_month = st.slider("Order Month", 1, 12, 6,
                            help="Month when order was placed")
    benefit_per_order = st.number_input("Benefit Per Order ($)",
                                         min_value=-500.0,
                                         max_value=1000.0,
                                         value=50.0,
                                         step=10.0)
    order_item_quantity = st.number_input("Order Item Quantity",
                                           min_value=1,
                                           max_value=10,
                                           value=2)

col3, col4 = st.columns(2)
with col3:
    customer_segment = st.selectbox("Customer Segment", [
        "Consumer", "Corporate", "Home Office"
    ])
    market = st.selectbox("Market", [
        "Europe", "LATAM", "Pacific Asia", "USCA", "Africa"
    ])
with col4:
    order_item_discount_rate = st.slider("Discount Rate", 0.0, 1.0, 0.1, 0.05)
    product_price = st.number_input("Product Price ($)",
                                     min_value=5.0,
                                     max_value=2000.0,
                                     value=100.0,
                                     step=10.0)
st.markdown("<br>", unsafe_allow_html=True)
if st.button("🔍 PREDICT DELAY RISK"):
    input_data = {
        'Type_DEBIT': 0,
        'Type_PAYMENT': 0,
        'Type_TRANSFER': 0,
        'Benefit per order': benefit_per_order,
        'Sales per customer': product_price * order_item_quantity,
        'Category Id': 1,
        'Customer Id': 1,
        'Customer Zipcode': 10001.0,
        'Department Id': 1,
        'Latitude': 40.0,
        'Longitude': -75.0,
        'Order Customer Id': 1,
        'Order Item Cardprod Id': 1,
        'Order Item Discount': product_price * order_item_discount_rate,
        'Order Item Discount Rate': order_item_discount_rate,
        'Order Item Id': 1,
        'Order Item Product Price': product_price,
        'Order Item Profit Ratio': 0.3,
        'Order Item Quantity': order_item_quantity,
        'Sales': product_price * order_item_quantity,
        'Order Item Total': product_price * order_item_quantity,
        'Order Profit Per Order': benefit_per_order,
        'Product Category Id': 1,
        'Product Price': product_price,
        'Product Status': 0,
        'shipping_duration': 3,
        'delay_days': 0,
        'order_month': order_month,
        'order_year': 2024,
        'order_dayofweek': 1,
    }
    input_df = pd.DataFrame([input_data])
    ohe_cols = {
        'Shipping Mode': shipping_mode,
        'Customer Segment': customer_segment,
        'Market': market,
        'Order Region': order_region,
        'Category Name': category_name,
    }
    for col, val in ohe_cols.items():
        dummy_df = pd.get_dummies(pd.Series([val], name=col), prefix=col, drop_first=False)
        for dcol in dummy_df.columns:
            input_df[dcol] = dummy_df[dcol].values[0]
    for col in model_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[model_columns]
    input_df = input_df.fillna(0)

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0]

    prob_delayed = round(probability[1] * 100, 1)
    prob_ontime = round(probability[0] * 100, 1)

    if prediction == 1:
        st.markdown(f"""
        <div class="result-delayed">
            <div class="result-emoji"></div>
            <div class="result-label" style="color:#fc8181;">DELAYED</div>
            <div class="result-prob">Delay Probability: <strong style="color:#fc8181;">{prob_delayed}%</strong></div>
            <div class="prob-bar-container">
                <div style="background:#e53e3e; width:{prob_delayed}%; height:100%; border-radius:999px;"></div>
            </div>
            <div style="color:#a0aec0; font-size:0.85rem;">This shipment is at high risk of arriving late.</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-ontime">
            <div class="result-emoji"></div>
            <div class="result-label" style="color:#68d391;">ON TIME</div>
            <div class="result-prob">On-Time Probability: <strong style="color:#68d391;">{prob_ontime}%</strong></div>
            <div class="prob-bar-container">
                <div style="background:#38a169; width:{prob_ontime}%; height:100%; border-radius:999px;"></div>
            </div>
            <div style="color:#a0aec0; font-size:0.85rem;">This shipment is expected to arrive on schedule.</div>
        </div>
        """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">💡 Insights</div>', unsafe_allow_html=True)
    tips = []
    if shipping_mode in ["First Class", "Second Class"]:
        tips.append(" Faster shipping modes (First & Second Class) have historically higher delay rates than Standard Class.")
    if order_region in ["Western Europe", "Central America", "South America"]:
        tips.append(f"{order_region} is one of the regions with the highest delay rates in the dataset.")
    if order_month in [11, 12]:
        tips.append("🎄 Holiday months (Nov-Dec) tend to see increased order volumes which can contribute to delays.")
    if not tips:
        tips.append(" No major risk factors detected for this shipment configuration.")
    for tip in tips:
        st.markdown(f'<div class="info-card">{tip}</div>', unsafe_allow_html=True)
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center; color:#4a5568; font-size:0.8rem; font-family:'Space Mono', monospace;">
    CS 210 · Data Management for Data Science · Smit Madat (snm178)<br>
    Random Forest Classifier · AUC: 0.803 · 5-Fold CV F1: 0.704
</div>
""", unsafe_allow_html=True)
