# app.py

import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
from utils import prepare_recursive_forecast_input

# Custom page config
st.set_page_config(page_title="Retail Demand Forecasting", layout="centered")

# Custom CSS for colors
st.markdown("""
    <style>
    .main {
        background-color: #f2f2f2;
    }
    .block-container {
        background-color: #ffffff;
        padding: 2rem;
        border-radius: 10px;
    }
    .stButton>button {
        background-color: #d63384;
        color: white;
    }
    .stSelectbox>div>div>div {
        color: black;
    }
    </style>
""", unsafe_allow_html=True)

# Load model and data
model = joblib.load(r'C:\Users\Lenovo\OneDrive\Desktop\demand_forecast_app\model\rf_model.pkl')
df = pd.read_csv(r'C:\Users\Lenovo\OneDrive\Desktop\demand_forecast_app\data\demad_forecasting.csv', parse_dates=['date'])

# Title and intro
st.title("🛍️ Retail Demand Forecasting")
st.write("Predict sales for the next 7 days for any Store and Item combination.")

# Sidebar inputs
store_id = st.selectbox("Select Store ID", sorted(df['store'].unique()))
item_id = st.selectbox("Select Item ID", sorted(df['item'].unique()))

if st.button("🔮 Predict Sales"):
    future_dates, forecasts = prepare_recursive_forecast_input(df, model, store_id, item_id)

    if not forecasts:
        st.warning("Not enough historical data to make predictions.")
    else:
        result_df = pd.DataFrame({
            'date': future_dates,
            'predicted_sales': forecasts
        })

        result_df.to_csv(r"C:\Users\Lenovo\OneDrive\Desktop\demand_forecast_app\forecast_log.csv", index=False)

        st.subheader("📈 7-Day Forecast")
        st.write(result_df)

        fig = px.line(result_df, x='date', y='predicted_sales', title='Sales Forecast',
                      line_shape='linear', markers=True,
                      color_discrete_sequence=['#d63384'])  # dark pink
        st.plotly_chart(fig)
