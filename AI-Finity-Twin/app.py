# streamlit_app.py
"""
Streamlit app: Financial Digital Twin prototype + GROQ connector (Sanity)
Usage:
  1. Set env vars: SANITY_PROJECT_ID, SANITY_DATASET, SANITY_TOKEN (optional for public datasets)
  2. pip install -r requirements.txt
  3. streamlit run streamlit_app.py
"""

import os
import urllib.parse
import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import math

from sanity_groq import sanity_query

st.set_page_config(page_title="Digital Twin Accounting (Prototype)", layout="wide")

# ---------- Helpers ----------
def linear_forecast(series_values, periods=6):
    """Simple linear fit forecast (very lightweight)."""
    if len(series_values) < 2:
        return [series_values[-1]] * periods if series_values else [0]*periods
    x = np.arange(len(series_values))
    y = np.array(series_values)
    coeffs = np.polyfit(x, y, 1)
    slope, intercept = coeffs
    future_x = np.arange(len(series_values), len(series_values) + periods)
    forecast = intercept + slope * future_x
    return forecast.tolist()

def to_currency(x):
    return f"Rp {x:,.0f}"

# ---------- Sidebar: Sanity / GROQ config ----------
st.sidebar.title("Config Sanity (GROQ)")
project = st.sidebar.text_input("Project ID", os.getenv("SANITY_PROJECT_ID", "your_project_id"))
dataset = st.sidebar.text_input("Dataset", os.getenv("SANITY_DATASET", "production"))
token = st.sidebar.text_input("Token (optional)", os.getenv("SANITY_TOKEN", ""), type="password")

st.sidebar.markdown("---")
st.sidebar.write("Jika kamu belum punya Sanity, kamu bisa tetap pakai demo dataset lokal.")

# ---------- Fetch data ----------
use_demo = st.sidebar.checkbox("Use demo data (no Sanity)", value=True)

@st.cache_data(ttl=300)
def fetch_transactions_from_sanity(limit=200):
    # Example GROQ: fetch transactions with date, amount, category
    query = '*[_type == "transaction"]{_id, date, amount, category, note}[0...%d]' % limit
    res = sanity_query(project, dataset, query, token)
    if not res or 'result' not in res:
        return []
    return res['result']

def demo_transactions():
    np.random.seed(42)
    base_date = datetime.now() - timedelta(days=60)
    items = []
    for i in range(60):
        d = (base_date + timedelta(days=i)).strftime("%Y-%m-%d")
        income = 500000 + np.random.randn() * 80000 + max(0, 20000*math.sin(i/5))
        expense = 200000 + np.random.randn() * 40000
        items.append({"date": d, "amount": float(round(income,2)), "category": "income", "note": "sales"})
        items.append({"date": d, "amount": float(round(-abs(expense),2)), "category": "expense", "note": "operational"})
    return items

if use_demo:
    raw = demo_transactions()
else:
    raw = fetch_transactions_from_sanity()

if not raw:
    st.warning("Tidak ada data transaksi — gunakan demo data atau cek konfigurasi Sanity.")
    raw = demo_transactions()

df = pd.DataFrame(raw)
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date')

# ---------- Top row: KPIs ----------
st.title("Financial Digital Twin — Prototype")
col1, col2, col3, col4 = st.columns(4)
total_revenue = df.loc[df['amount']>0,'amount'].sum()
total_expense = -df.loc[df['amount']<0,'amount'].sum()
net = total_revenue - total_expense
col1.metric("Total Revenue (period)", to_currency(total_revenue))
col2.metric("Total Expense (period)", to_currency(total_expense))
col3.metric("Net", to_currency(net))
col4.metric("Days covered", f"{(df['date'].max() - df['date'].min()).days} days")

# ---------- Time series and forecast ----------
st.subheader("Cashflow Time Series & Forecast")
cashflow = df.groupby(df['date'].dt.date)['amount'].sum().reset_index()
cashflow.columns = ['date', 'amount']
cashflow = cashflow.sort_values('date')
st.line_chart(cashflow.set_index('date')['amount'])

periods = st.slider("Forecast horizon (days)", 7, 90, 30)
series_vals = cashflow['amount'].tolist()
forecast_vals = linear_forecast(series_vals, periods=periods)
future_dates = [cashflow['date'].max() + timedelta(days=i+1) for i in range(periods)]
forecast_df = pd.DataFrame({"date": future_dates, "forecast": forecast_vals})
forecast_df = forecast_df.set_index('date')
st.line_chart(pd.concat([cashflow.set_index('date')['amount'], forecast_df['forecast']], axis=1).fillna(method='ffill'))

# ---------- Digital Twin simulator ----------
st.subheader("Digital Twin Simulator (Quick What-If)")
st.markdown("Ubah parameter untuk melihat proyeksi pendapatan netto.")

col_a, col_b, col_c = st.columns(3)
price_mul = col_a.slider("Price multiplier (simulate price change)", 0.5, 1.5, 1.0, 0.01)
volume_mul = col_b.slider("Volume multiplier (simulate demand change)", 0.5, 1.5, 1.0, 0.01)
cost_mul = col_c.slider("Cost multiplier (simulate supplier cost change)", 0.7, 1.5, 1.0, 0.01)

# compute baseline daily avg revenue & expense
avg_income = df.loc[df['amount']>0,'amount'].abs().mean() if not df.loc[df['amount']>0].empty else 0
avg_expense = df.loc[df['amount']<0,'amount'].abs().mean() if not df.loc[df['amount']<0].empty else 0

sim_revenue = avg_income * price_mul * volume_mul
sim_cost = avg_expense * cost_mul
sim_net = sim_revenue - sim_cost

st.write("Baseline avg income/day:", to_currency(avg_income))
st.write("Baseline avg expense/day:", to_currency(avg_expense))
st.metric("Simulated revenue/day", to_currency(sim_revenue))
st.metric("Simulated cost/day", to_currency(sim_cost))
st.metric("Simulated net/day", to_currency(sim_net))

# quick text suggestion (predictive assistant minimal)
if sim_net < (avg_income - avg_expense) * 0.5:
    st.warning("Warning: projected net turun >50%. Rekomendasi: cek cost structure & pertimbangkan menaikkan harga.")
else:
    st.success("Proyeksi net terlihat stabil.")

# ---------- Data table & raw GROQ query ----------
st.subheader("Data transaksi (sample)")
st.dataframe(df.head(200))

st.expander("Run custom GROQ query (advanced)"):
    user_query = st.text_area("GROQ query", value='*[_type == "transaction"]{_id, date, amount, category, note}[0...50]')
    if st.button("Run GROQ"):
        if use_demo:
            st.info("Mode demo aktif — tidak menjalankan GROQ. Matikan 'Use demo data' di sidebar.")
        else:
            res = sanity_query(project, dataset, user_query, token)
            st.json(res)

st.markdown("---")
st.caption("Prototype: lightweight Digital Twin + GROQ connector. Untuk produksi: tambahkan auth, validasi, engine ML, dan monitoring.")
