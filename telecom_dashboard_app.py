# telecom_dashboard_app.py
try:
    import streamlit as st
except ModuleNotFoundError:
    raise ImportError("The 'streamlit' package is required to run this dashboard. Please install it using 'pip install streamlit'.")

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import IsolationForest

# Set page configuration
st.set_page_config(page_title="Cellular Network Dashboard", layout="wide")

# Load data
def load_data():
    st.sidebar.header("1. Load Dataset")
    data_file = st.sidebar.file_uploader("Upload train.csv from Kaggle Dataset", type=["csv"])
    if data_file is not None:
        df = pd.read_csv(data_file)
        return df
    else:
        st.sidebar.warning("Please upload a CSV file.")
        return None

# Clean data
def clean_data(df):
    df['Tower ID'] = df['Tower ID'].astype('category')
    df['User ID'] = df['User ID'].astype('category')
    if 'Timestamp' in df.columns:
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
    for col in ['Environment', 'Call Type', 'Incoming/Outgoing']:
        if col in df.columns:
            df[col] = df[col].astype('category')
    # Map environment
    env_map = {'open': 'rural', 'home': 'indoor', 'suburban': 'suburban', 'urban': 'urban'}
    df['Env_Std'] = df['Environment'].map(env_map)
    df['Location_Type'] = df['Env_Std'].apply(lambda x: 'indoor' if x == 'indoor' else 'outdoor')
    return df

# Summarize data
def summarize_data(df):
    st.subheader("📊 Quick Network Summary")
    avg_signal = df['Signal Strength (dBm)'].mean()
    weak_areas = len(df[df['Signal Strength (dBm)'] < -95])
    worst_tower = df.groupby('Tower ID')['Signal Strength (dBm)'].mean().idxmin()
    best_env = df.groupby('Environment')['SNR'].mean().idxmax()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Avg Signal Strength", f"{avg_signal:.1f} dBm")
    col2.metric("Critical Areas", f"{weak_areas} locations")
    col3.metric("Worst Tower", worst_tower)
    col4.metric("Best Environment", best_env)

# Filters
@st.cache_data

def filter_data(df, env_types, call_types, signal_range):
    df_filtered = df[
        df['Environment'].isin(env_types) &
        df['Call Type'].isin(call_types) &
        df['Signal Strength (dBm)'].between(signal_range[0], signal_range[1])
    ]
    return df_filtered

# Generate plots
def generate_charts(df):
    st.subheader("📈 Call Distribution")
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.countplot(data=df, x='Environment', hue='Call Type', palette='Set2', ax=ax)
    st.pyplot(fig)

    st.subheader("📉 Signal Strength vs. SNR")
    fig2 = px.scatter(df, x='Signal Strength (dBm)', y='SNR', color='Environment')
    st.plotly_chart(fig2)

# Outlier detection

def detect_outliers(df):
    st.subheader("🚨 Outlier Detection")
    df['SNR_Outlier'] = (df['SNR'] < 10) | (df['SNR'] > 40)
    df['Distance_Outlier'] = (df['Distance to Tower (km)'] > 30) | (df['Distance to Tower (km)'] < 0.1)
    df['ML_Outlier'] = IsolationForest(contamination=0.05).fit_predict(df[['SNR', 'Distance to Tower (km)']].abs()) == -1
    df['Total_Outlier'] = df['SNR_Outlier'] | df['Distance_Outlier'] | df['ML_Outlier']
    st.write(f"Total Outliers Detected: {df['Total_Outlier'].sum()}")
    return df

# MAIN LOGIC
st.title("📡 Cellular Network Performance Dashboard")
df = load_data()
if df is not None:
    df = clean_data(df)

    with st.sidebar:
        st.header("2. Filters")
        env_options = st.multiselect("Environment Types", options=df['Environment'].unique(), default=df['Environment'].unique())
        call_options = st.multiselect("Call Types", options=df['Call Type'].unique(), default=df['Call Type'].unique())
        signal_slider = st.slider("Signal Strength Range (dBm)", -120, -30, (-100, -60))

    df_filtered = filter_data(df, env_options, call_options, signal_slider)

    summarize_data(df_filtered)
    generate_charts(df_filtered)
    detect_outliers(df_filtered)
else:
    st.info("Upload a CSV file to begin analysis.")
