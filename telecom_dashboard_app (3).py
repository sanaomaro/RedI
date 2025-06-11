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
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Cellular Network Dashboard", layout="wide")

# Load data
def load_data():
    st.sidebar.header("1. Load Dataset")
    data_file = st.sidebar.file_uploader("Upload train.csv from Kaggle Dataset", type=["csv"])
    if data_file is not None:
        df = pd.read_csv(data_file)

        # Synthesize realistic 'Transmitted Data (MB)' and 'Noise (dB)' columns
       # np.random.seed(42)
       # df['Transmitted Data (MB)'] = np.abs(np.random.normal(loc=20, scale=10, size=len(df)))
       # df['Noise (dB)'] = np.abs(np.random.normal(loc=5, scale=2, size=len(df)))
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

# Chart visualizations
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
    df['ML_Outlier'] = IsolationForest(contamination=0.05, random_state=42).fit_predict(df[['SNR', 'Distance to Tower (km)']].abs()) == -1
    df['Total_Outlier'] = df['SNR_Outlier'] | df['Distance_Outlier'] | df['ML_Outlier']
    st.write(f"Total Outliers Detected: {df['Total_Outlier'].sum()}")
    return df

# Propagation Model using distance bands

def propagation_model(df):
    st.subheader("📐 Propagation Modeling by Distance Band")
    df['Distance Band'] = pd.cut(
        df['Distance to Tower (km)'],
        bins=[0, 1, 3, 6, 10, np.inf],
        labels=['very close', 'close', 'mid', 'far', 'very far']
    )
    st.write("### Median SNR per Distance Band")
    st.dataframe(df.groupby('Distance Band')['SNR'].median())

# Tower stats overview

def tower_stats(df):
    st.subheader("📡 Tower Statistics")
    tower_df = df.groupby('Tower ID').agg(
        avg_signal=('Signal Strength (dBm)', 'mean'),
        max_distance=('Distance to Tower (km)', 'max'),
        min_signal=('Signal Strength (dBm)', 'min'),
        max_signal=('Signal Strength (dBm)', 'max'),
        call_count=('User ID', 'count')
    ).reset_index()
    st.dataframe(tower_df)

# Regression Analysis

def regression_analysis(df):
    st.subheader("📊 Regression: Signal vs Distance + Attenuation")
    df = df.dropna(subset=['Distance to Tower (km)', 'Attenuation', 'Signal Strength (dBm)'])
    X = df[['Distance to Tower (km)', 'Attenuation']]
    y = df['Signal Strength (dBm)']
    model = LinearRegression().fit(X, y)
    st.write(f"Signal loss per km: {model.coef_[0]:.2f} dBm/km")
    st.write(f"Signal loss per dB attenuation: {model.coef_[1]:.2f} dBm")
    fig = px.scatter(df, x='Distance to Tower (km)', y='Signal Strength (dBm)', color='Environment')
    st.plotly_chart(fig)

# Time series exploration

def snr_time_series(df):
    st.subheader("⏱️ SNR Trend Over Time")
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
    df.set_index('Timestamp', inplace=True)
    ts = df['SNR'].resample('D').mean()
    fig, ax = plt.subplots(figsize=(10, 4))
    ts.plot(ax=ax)
    ax.set_title("Daily Average SNR")
    st.pyplot(fig)
    df.reset_index(inplace=True)

# Predictive model using telecom standards and log decay

def predict_snr(df):
    st.subheader("🔮 SNR Prediction using Random Forest")
    features = ['Distance to Tower (km)', 'Attenuation', 'Signal Strength (dBm)', 'Transmitted Data (MB)', 'Noise (dB)']
    df = df.dropna(subset=features + ['SNR'])
    X = df[features]
    y = df['SNR']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    y_pred = rf_model.predict(X_test)
    st.write(f"R² Score: {r2_score(y_test, y_pred):.3f}")
    st.write(f"MSE: {mean_squared_error(y_test, y_pred):.2f}")
    fig = px.scatter(x=y_test, y=y_pred, labels={'x': 'True SNR', 'y': 'Predicted SNR'}, title="Predicted vs True SNR")
    fig.add_shape(type='line', x0=y_test.min(), y0=y_test.min(), x1=y_test.max(), y1=y_test.max(), line=dict(color='Red', dash='dash'))
    st.plotly_chart(fig)

# Telecom standards documentation

def show_telecom_standards():
    with st.expander("📚 Telecom Assumptions & Standards"):
        st.markdown("""
        - **Transmitted Power:** Simulated between 5 and 50 MB to represent typical mobile data sessions.
        - **Noise Power:** Ranges from 90–120 dB, based on industry values for background RF noise.
        - **SNR (Signal-to-Noise Ratio):** Predicted using Random Forests from signal strength, distance, attenuation, transmitted data, and noise.
        - **SNR Benchmarking:** Good SNR for LTE is 20–30 dB; below 10 dB is poor.
        - **Outliers:** Detected using traditional thresholds and ML (Isolation Forest).
        """)

# Export buttons

def export_data(df):
    st.subheader("📥 Export Your Analysis")
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("Download Filtered CSV", csv, file_name='filtered_data.csv', mime='text/csv')

# MAIN LOGIC
st.title("📡 Cellular Network Performance Dashboard")
df = load_data()
if df is not None:
    df = clean_data(df)
    with st.sidebar:
        st.header("2. Filters")
        env_options = st.multiselect("Environment Types", options=df['Environment'].unique(), default=list(df['Environment'].unique()))
        call_options = st.multiselect("Call Types", options=df['Call Type'].unique(), default=list(df['Call Type'].unique()))
        signal_slider = st.slider("Signal Strength Range (dBm)", -120, -30, (-100, -60))
    show_telecom_standards()
    df_filtered = df[
        df['Environment'].isin(env_options) &
        df['Call Type'].isin(call_options) &
        df['Signal Strength (dBm)'].between(signal_slider[0], signal_slider[1])
    ]
    summarize_data(df_filtered)
    generate_charts(df_filtered)
    detect_outliers(df_filtered)
    propagation_model(df_filtered)
    tower_stats(df_filtered)
    regression_analysis(df_filtered)
    snr_time_series(df_filtered)
    predict_snr(df_filtered)
    export_data(df_filtered)
else:
    st.info("Upload a CSV file to begin analysis.")
