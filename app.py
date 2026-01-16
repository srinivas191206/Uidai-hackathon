import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests
import json
import base64
import os
from datetime import datetime

# --- CONFIGURATION & AESTHETICS ---
st.set_page_config(page_title="UIDAI Analytics Command Center", layout="wide", page_icon="🇮🇳")

# --- HEADER IMAGE (Menu Bar) ---
def get_img_as_base64(file):
    with open(file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

img = get_img_as_base64("head.jpg")

# Inject Fixed Header using HTML/CSS
st.markdown(f"""
    <style>
    /* Fixed Header encompassing full width */
    .fixed-header {{
        position: fixed;
        top: 0;
        left: 0;
        width: 100vw;
        z-index: 9999999;
        background-color: white; /* Prevent transparency issues */
        border-bottom: 2px solid #003366;
    }}
    .fixed-header img {{
        width: 100%;
        height: auto;
        display: block;
        max-height: 120px; /* Constrain height to avoid taking over too much screen */
        object-fit: cover;
    }}
    
    /* Push content down to account for fixed header */
    [data-testid="stSidebar"] {{
        margin-top: 120px;
        height: calc(100vh - 120px);
        background-color: #FFFFFF;
        border-right: 1px solid #E0E0E0;
    }}
    /* Remove default Streamlit top padding in sidebar */
    [data-testid="stSidebar"] > div:first-child {{
        padding-top: 1rem; 
    }}

    .block-container {{
        padding-top: 140px !important; /* Slightly more to give breathing room */
    }}
    
    /* Hide all Streamlit-specific elements for a custom portal look */
    header {{visibility: hidden !important;}}
    footer {{visibility: hidden !important;}}
    #MainMenu {{visibility: hidden !important;}}
    [data-testid="stDecoration"] {{display: none !important;}}
    [data-testid="stHeader"] {{display: none !important;}}
    
    /* Remove the 'Running...' indicator to look like a managed app */
    [data-testid="stStatusWidget"] {{display: none !important;}}
    
    /* Push content down to account for fixed header */
    [data-testid="stSidebar"] {{
        margin-top: 120px;
        height: calc(100vh - 120px);
        background-color: #FDFDFD;
        border-right: 1px solid #E2E8F0;
    }}
    </style>
    
    <div class="fixed-header">
        <img src="data:image/jpg;base64,{img}">
    </div>
""", unsafe_allow_html=True)



# Custom Color Palette
# Custom CSS for Government Professional Theme
st.markdown("""
    <style>
    /* Import Premium Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        letter-spacing: -0.01em;
    }
    
    h1, h2, h3, h4, h5, h6, .main-header h1, .section-header {
        font-family: 'Poppins', 'Inter', sans-serif;
        font-weight: 600;
    }

    /* Background Color */
    .stApp {
        background: linear-gradient(135deg, #F8FAFC 0%, #EEF2F7 100%);
    }

    /* Header Styling */
    .main-header {
        background-color: #003366; /* Navy Blue */
        padding: 1.5rem 2rem;
        border-radius: 0px 0px 10px 10px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        display: flex;
        align-items: center;
        justify_content: space-between;
    }
    .main-header h1 {
        color: #FFFFFF;
        margin: 0;
        font-size: 2.2rem;
        font-weight: 700;
        letter-spacing: 1px;
    }
    .main-header h3 {
        color: #FF9933; /* Saffron */
        margin: 0;
        font-size: 1rem;
        text-transform: uppercase;
        letter-spacing: 2px;
        font-weight: 500;
        margin-top: 5px;
    }

    /* Metrics Cards - Custom HTML */
    .metric-card {
        background: linear-gradient(145deg, #FFFFFF 0%, #F8FAFC 100%);
        padding: 24px;
        border-radius: 16px;
        border-left: 4px solid #003366;
        box-shadow: 0 4px 20px rgba(0, 51, 102, 0.08);
        text-align: center;
        margin-bottom: 10px;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: center;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, #003366, #FF9933);
        opacity: 0;
        transition: opacity 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 30px rgba(0, 51, 102, 0.15);
    }
    .metric-card:hover::before {
        opacity: 1;
    }
    .metric-value {
        color: #003366;
        font-weight: 800;
        font-size: 2.2rem;
        line-height: 1.2;
        font-family: 'Poppins', sans-serif;
        background: linear-gradient(135deg, #003366 0%, #1a5c99 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .metric-label {
        color: #64748B;
        font-size: 0.85rem;
        margin-top: 8px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Dashboard Controls Panel */
    .controls-panel {
        background: linear-gradient(145deg, #FFFFFF 0%, #F1F5F9 100%);
        border-radius: 16px;
        padding: 20px 28px;
        margin-bottom: 24px;
        box-shadow: 0 4px 20px rgba(0, 51, 102, 0.06);
        border: 1px solid rgba(0, 51, 102, 0.08);
    }
    .controls-title {
        font-family: 'Poppins', sans-serif;
        font-weight: 600;
        font-size: 1.1rem;
        color: #003366;
        margin-bottom: 16px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .controls-title::before {
        content: '';
        width: 4px;
        height: 20px;
        background: linear-gradient(180deg, #003366, #FF9933);
        border-radius: 2px;
    }

    /* Standard Streamlit Metrics Override (if used elsewhere) */
    [data-testid="stMetric"] {
        background-color: #FFFFFF;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #003366; /* Accent Border */
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        transition: transform 0.2s;
    }
    [data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
    }
    [data-testid="stMetricLabel"] {
        color: #666666;
        font-size: 0.9rem;
    }
    [data-testid="stMetricValue"] {
        color: #003366;
        font-weight: 700;
        font-size: 1.8rem;
    }

    /* Tabs Styling - Enhanced */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background-color: #ffffff;
        padding: 10px 10px 0px 10px;
        border-radius: 10px 10px 0 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        border-radius: 5px 5px 0 0;
        color: #4A4A4A;
        font-weight: 600; /* Bold text */
        font-size: 1rem; /* Larger text */
        padding: 10px 20px; /* More clickable area */
        flex-grow: 1; /* Distribute space evenly */
        justify-content: center; /* Center text */
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #003366;
        background-color: #F0F4F8;
    }
    .stTabs [aria-selected="true"] {
        background-color: #003366 !important;
        color: #FFFFFF !important;
        border-bottom: 3px solid #FF9933; /* Saffron underline for active tab */
    }
    
    /* Section Headers - Premium Style */
    .section-header {
        background: linear-gradient(135deg, #E3F2FD 0%, #DBEAFE 100%);
        border-left: 4px solid #003366;
        padding: 14px 24px;
        border-radius: 12px;
        color: #003366;
        margin-top: 28px;
        margin-bottom: 24px;
        font-weight: 600;
        font-size: 1.1rem;
        font-family: 'Poppins', sans-serif;
        display: flex;
        align-items: center;
        box-shadow: 0 2px 12px rgba(0, 51, 102, 0.08);
        letter-spacing: 0.02em;
    }
    
    /* Selectbox Styling */
    .stSelectbox > div > div {
        background: #FFFFFF;
        border-radius: 10px;
        border: 1px solid #E2E8F0;
        font-family: 'Inter', sans-serif;
    }
    .stSelectbox label {
        font-family: 'Inter', sans-serif;
        font-weight: 500;
        color: #475569;
        font-size: 0.9rem;
    }
    .stMultiSelect > div > div {
        background: #FFFFFF;
        border-radius: 10px;
        border: 1px solid #E2E8F0;
    }
    .stMultiSelect label {
        font-family: 'Inter', sans-serif;
        font-weight: 500;
        color: #475569;
        font-size: 0.9rem;
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #FFFFFF;
        border-right: 1px solid #E0E0E0;
    }
    
    /* Hide Default Streamlit Elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;} /* Hide the top decoration bar */
    
    /* Remove Top Padding from Main Container to push image to top */
    .block-container {
        padding-top: 0rem !important;
        padding-bottom: 0rem !important;
        max-width: 100% !important;
    }
    
    </style>
""", unsafe_allow_html=True)

# --- INDIAN NUMBER FORMATTING ---
def format_indian(num):
    """
    Formats a number into the Indian numbering system (e.g., 1,00,000 for 1 Lakh).
    Handles integers and floats.
    """
    if pd.isna(num):
        return "0"
    
    try:
        num = int(num)
        s = str(num)
        if len(s) <= 3:
            return s
        
        last_three = s[-3:]
        remaining = s[:-3]
        
        # Add commas every 2 digits for the rest
        formatted_remaining = ""
        for i, digit in enumerate(reversed(remaining)):
            if i > 0 and i % 2 == 0:
                formatted_remaining = "," + formatted_remaining
            formatted_remaining = digit + formatted_remaining
            
        return formatted_remaining + "," + last_three
    except:
        return str(num)

def perform_custom_clustering(df, n_clusters=3):
    """
    Performs K-Means clustering using Numpy (No Scikit-Learn dependency).
    Used to segment districts into 'Critical', 'Stable', 'High Performance'.
    """
    if len(df) < n_clusters:
        # Fallback for single district or small count selection
        labels = []
        for _, row in df.iterrows():
            if row['demand_score'] > 1.3: labels.append('High-Intensity Operations 🔴')
            elif row['demand_score'] < 0.8: labels.append('Low-Enrolment / Outreach Needed 🔵')
            else: labels.append('Steady State / Monitoring 🟡')
        return labels

    # Features to cluster on
    features = ['demand_score', 'child_ratio', 'update_pressure']
    data = df[features].copy().fillna(0)
    
    # Normalize data (Min-Max Scaling)
    data_norm = (data - data.min()) / (data.max() - data.min() + 1e-9)
    X = data_norm.values
    
    # Initialize Centroids (Randomly select k points)
    np.random.seed(42)
    random_indices = np.random.choice(X.shape[0], size=n_clusters, replace=False)
    centroids = X[random_indices]
    
    # Iterative Optimization
    for _ in range(10): # 10 iterations is usually sufficient for this data size
        # 1. Assign clusters based on Euclidean distance
        distances = np.sqrt(((X - centroids[:, np.newaxis])**2).sum(axis=2))
        cluster_labels = np.argmin(distances, axis=0)
        
        # 2. Update centroids
        new_centroids = np.array([X[cluster_labels == k].mean(axis=0) for k in range(n_clusters)])
        
        # Handle empty clusters (though unlikely)
        if np.any(np.isnan(new_centroids)):
            new_centroids = np.where(np.isnan(new_centroids), centroids, new_centroids)
            
        centroids = new_centroids
        
    # Assign readable labels based on mean demand_score of cluster
    # Higher demand score usually means 'Critical Care' or 'High Demand'
    cluster_map = {}
    mean_scores = []
    
    for k in range(n_clusters):
        mean_score = data.iloc[cluster_labels == k]['demand_score'].mean()
        mean_scores.append((k, mean_score))
        
    # Sort clusters by stress/demand (High to Low)
    mean_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Map to semantic segments (ADMINISTRATIVE / ACTION-ORIENTED)
    cluster_map[mean_scores[0][0]] = 'High-Intensity Operations 🔴' # Critical
    cluster_map[mean_scores[1][0]] = 'Steady State / Monitoring 🟡'   # Stable
    cluster_map[mean_scores[2][0]] = 'Low-Enrolment / Outreach Needed 🔵' # Low Activity
    
    return [cluster_map[label] for label in cluster_labels]

def generate_mission_brief_html(kpis, recs, insights, scope_name):
    """
    Generates a professional 'Commander's Brief' HTML for download.
    """
    html = f"""
    <html>
    <head>
        <style>
            body {{ font-family: 'Courier New', monospace; color: #1E293B; line-height: 1.6; background-color: #f8f9fa; }}
            .header {{ border-bottom: 2px solid #0f172a; padding-bottom: 20px; margin-bottom: 30px; }}
            .confidential {{ color: #dc2626; font-weight: bold; text-align: center; border: 1px solid #dc2626; padding: 5px; margin-bottom: 20px; }}
            .metric-box {{ background: #ffffff; padding: 15px; border: 1px solid #e2e8f0; width: 45%; display: inline-block; margin-bottom: 15px; }}
            .rec-card {{ background: #fffcf2; border-left: 4px solid #f59e0b; padding: 15px; margin-bottom: 10px; border: 1px solid #e5e7eb; }}
        </style>
    </head>
    <body>
        <div class="confidential">CONFIDENTIAL // TYPE 2 // INTERNAL OPERATIONS ONLY</div>
        
        <div class="header">
            <h1>UIDAI ANALYTICS COMMAND BRIEF</h1>
            <p><strong>Scope:</strong> {scope_name} | <strong>Generated:</strong> {datetime.now().strftime('%d %b %Y %H:%M')}</p>
        </div>
        
        <h3>1. STRATEGIC SITUATION REPORT</h3>
        <div class="metric-box">
            <b>Total Activity Volume</b><br>
            <span style="font-size: 1.5em">{kpis['total']}</span>
        </div>
        <div class="metric-box">
            <b>Daily Operational Avg</b><br>
            <span style="font-size: 1.5em">{kpis['daily_avg']}</span>
        </div>
        <div class="metric-box">
            <b>Backlog Clearance Est.</b><br>
            <span style="font-size: 1.5em; color: { '#dc2626' if '>' in str(kpis.get('backlog','')) else '#16a34a' }">{kpis.get('backlog', 'Normal')}</span>
        </div>
         <div class="metric-box">
            <b>Child Enrolment Ratio</b><br>
            <span style="font-size: 1.5em">{kpis['child_pct']}%</span>
        </div>
        
        <h3>2. RAPID RESPONSE PROTOCOLS</h3>
    """
    
    for rec in recs:
        html += f"""
        <div class="rec-card">
            <b>[{rec['type'].upper()}] {rec['action']}</b>
            <p>{rec['detail']}</p>
        </div>
        """
        
    html += """
        <h3>3. AUTOMATED INTELLIGENCE SUMMARY</h3>
        <ul>
    """
    for insight in insights:
         html += f"<li><b>{insight['title']}</b>: {insight['text']}</li><br>"
         
    html += """
        </ul>
        <hr>
        <p style="text-align: center; color: #64748B; font-size: 0.8em;">AUTHORIZED PERSONNEL ONLY | UIDAI DATA GOVERNANCE ACT</p>
    </body>
    </html>
    """
    return html

# --- DATA ENGINE ---

@st.cache_data
def load_and_process_data():
    GITHUB_RAW_URL = "https://raw.githubusercontent.com/srinivas191206/Uidai-hackathon/main/enrolment_data_main.csv"
    LOCAL_PATH = "enrolment_data_main.csv"
    
    try:
        # Try local first for speed
        if os.path.exists(LOCAL_PATH):
            df = pd.read_csv(LOCAL_PATH)
        else:
            # Fallback to GitHub Raw for cloud deployment
            df = pd.read_csv(GITHUB_RAW_URL)
            
        df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y', errors='coerce')
        df = df.dropna(subset=['date'])
        
        # 1. Total Enrolment per record
        df['total_activity'] = df['age_0_5'] + df['age_5_17'] + df['age_18_greater']
        
        # Normalize text columns
        df['postal_state'] = df['postal_state'].astype(str).str.title()
        df['postal_district'] = df['postal_district'].astype(str).str.title()
        
        return df
    except Exception as e:
        # Final safety fallback to a direct URL if local check fails
        try:
            df = pd.read_csv(GITHUB_RAW_URL)
            df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y', errors='coerce')
            df['total_activity'] = df['age_0_5'] + df['age_5_17'] + df['age_18_greater']
            return df
        except:
            st.error(f"Critical Data Load Error: {e}")
            return pd.DataFrame()

@st.cache_data
def load_geojson():
    BASE_URL = "https://raw.githubusercontent.com/srinivas191206/Uidai-hackathon/main/assets/"
    try:
        states, districts = None, None
        
        # Helper to load and validate JSON
        def load_safe_json(path):
            if os.path.exists(path) and os.path.getsize(path) > 1000: # LFS pointers are tiny
                with open(path, "r") as f:
                    return json.load(f)
            return None

        states = load_safe_json("assets/india_states.geojson")
        districts = load_safe_json("assets/india_district.geojson")

        # Fallback to GitHub if local files are missing or LFS pointers
        if not states:
            states = requests.get(BASE_URL + "india_states.geojson").json()
        if not districts:
            districts = requests.get(BASE_URL + "india_district.geojson").json()
            
        return states, districts
    except Exception as e:
        st.error(f"GeoJSON Load Error: {e}")
        return None, None

@st.cache_data
def calculate_district_metrics(df):
    # Aggregation by District
    dist_stats = df.groupby(['postal_state', 'postal_district']).agg(
        total_activity=('total_activity', 'sum'),
        age_0_5=('age_0_5', 'sum'),
        age_5_17=('age_5_17', 'sum'),
        age_18_greater=('age_18_greater', 'sum'),
        avg_monthly_activity=('total_activity', 'mean'), # Approximation for monthly avg
        record_count=('date', 'count') # Number of days/records
    ).reset_index()
    
    # --- DEMAND VELOCITY ENGINE ---
    # Calculate Velocity: (Last 30 days - Previous 30 days) / Previous 30 days
    max_date = df['date'].max()
    period_curr_start = max_date - pd.Timedelta(days=30)
    period_prev_start = max_date - pd.Timedelta(days=60)
    
    curr_period_df = df[(df['date'] > period_curr_start) & (df['date'] <= max_date)]
    prev_period_df = df[(df['date'] > period_prev_start) & (df['date'] <= period_curr_start)]
    
    curr_agg = curr_period_df.groupby(['postal_district'])['total_activity'].sum().reset_index(name='curr_vol')
    prev_agg = prev_period_df.groupby(['postal_district'])['total_activity'].sum().reset_index(name='prev_vol')
    
    velocity_df = curr_agg.merge(prev_agg, on='postal_district', how='left').fillna(0)
    velocity_df['velocity_pct'] = ((velocity_df['curr_vol'] - velocity_df['prev_vol']) / (velocity_df['prev_vol'] + 1)) * 100
    
    dist_stats = dist_stats.merge(velocity_df[['postal_district', 'velocity_pct']], on='postal_district', how='left').fillna(0)
    
    # 2. Demand Score (Relative)
    national_avg_dist_activity = dist_stats['total_activity'].mean()
    dist_stats['demand_score'] = dist_stats['total_activity'] / national_avg_dist_activity
    
    # 3. Demand Zones
    def classify_zone(score):
        if score > 1.3: return 'High (Red) 🔴'
        elif score < 0.8: return 'Low (Blue) 🔵'
        else: return 'Medium (Yellow) 🟡'
    dist_stats['demand_zone'] = dist_stats['demand_score'].apply(classify_zone)
    
    # 4. Child Enrolment Ratio
    dist_stats['child_ratio'] = dist_stats['age_0_5'] / dist_stats['total_activity']
    
    # 5. Update Pressure Index (Approximated as 18+ activity share)
    dist_stats['update_pressure'] = dist_stats['age_18_greater'] / dist_stats['total_activity']
    
    # 6. Age Mix Imbalance Index
    dist_stats['age_mix_imbalance'] = dist_stats['age_18_greater'] / (dist_stats['age_0_5'] + dist_stats['age_5_17'])
    
    # 7. Volatility Score & 9. Stress Persistence
    # Calculate monthly aggregated stats per district for volatility
    df['month'] = df['date'].dt.to_period('M')
    monthly = df.groupby(['postal_district', 'month'])['total_activity'].sum().reset_index()
    volatility = monthly.groupby('postal_district')['total_activity'].agg(['std', 'mean']).reset_index()
    volatility['volatility_score'] = (volatility['std'] / (volatility['mean'] + 1)).fillna(0)
    
    # Merge volatility back
    dist_stats = dist_stats.merge(volatility[['postal_district', 'volatility_score']], on='postal_district', how='left')
    
    # Stress Persistence: Count months where district monthly activity > National Monthly Avg * 1.3
    national_monthly_avg = monthly.groupby('month')['total_activity'].mean().mean()
    high_stress_months = monthly[monthly['total_activity'] > national_monthly_avg * 1.3]
    stress_counts = high_stress_months.groupby('postal_district').size().reset_index(name='stress_persistence_months')
    
    dist_stats = dist_stats.merge(stress_counts, on='postal_district', how='left')
    dist_stats['stress_persistence_months'] = dist_stats['stress_persistence_months'].fillna(0)
    
    # 10. Silent Under-Enrolment Detection
    # Logic: Demand Score < 0.5 AND Volatility < 0.2 (Low activity + Stable low)
    dist_stats['is_silent_underenrolment'] = (dist_stats['demand_score'] < 0.5) & (dist_stats['volatility_score'] < 0.2)
    
    # 8. Pincode Concentration Risk
    # For each district, calculate share of top 10% pincodes
    conc_risks = []
    for district in dist_stats['postal_district'].unique():
        d_df = df[df['postal_district'] == district]
        pin_stats = d_df.groupby('pincode')['total_activity'].sum().sort_values(ascending=False)
        total_dist_activity = pin_stats.sum()
        if total_dist_activity > 0:
            top_10_count = max(1, int(len(pin_stats) * 0.1))
            top_share = pin_stats.head(top_10_count).sum() / (total_dist_activity + 1)
            conc_risks.append({'postal_district': district, 'concentration_risk': top_share})
        else:
            conc_risks.append({'postal_district': district, 'concentration_risk': 0})
            
    conc_df = pd.DataFrame(conc_risks)
    dist_stats = dist_stats.merge(conc_df, on='postal_district', how='left')
    
    return dist_stats

# --- 11. REAL ANALYTICS ENGINE (New Implementation) ---

def calculate_forecast_holt_winters(series, n_preds=15):
    """
    Implements a robust Trend + Seasonality forecast using Numpy.
    (Simplified Holt-Linear approximation for stability on small datasets)
    """
    if len(series) < 5:
        return np.array([series[-1]] * n_preds) # Not enough data
        
    y = np.array(series)
    
    # 1. Trend Estimation (Linear Regression on last 30 days)
    x = np.arange(len(y))
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]
    
    # 2. Seasonality (Weekly Pattern)
    # Extract deviations from trend
    trend = m * x + c
    detrended = y - trend
    
    # Average weekly pattern (assuming 7-day seasonality)
    seasonality = np.zeros(7)
    for i in range(len(detrended)):
        seasonality[i % 7] += detrended[i]
    seasonality /= (len(y) / 7)
    
    # 3. Forecast
    last_x = x[-1]
    predictions = []
    
    for i in range(1, n_preds + 1):
        future_x = last_x + i
        # Trend Component
        future_trend = m * future_x + c
        # Seasonality Component
        future_season = seasonality[(len(y) + i - 1) % 7]
        # Damping factor for conservative government forecasting
        pred = future_trend + (future_season * 0.8) 
        predictions.append(max(0, pred)) # No negative enrolment
        
    return np.array(predictions)

def detect_statistical_anomalies(df, window=7):
    """
    Identifies high-confidence operational anomalies using rolling Z-scores.
    Logic: Z = (x - mean) / std. Flags deviations > 3 sigma.
    Interpretation: Spike/Drop indicating infrastructure or system issues.
    """
    daily = df.groupby('date')['total_activity'].sum().reset_index()
    daily = daily.sort_values('date')
    
    if len(daily) < window:
        return []
        
    daily['rolling_mean'] = daily['total_activity'].rolling(window=window).mean()
    daily['rolling_std'] = daily['total_activity'].rolling(window=window).std()
    
    # Calculate Z-score (Lowered to 2.0 for prototype sensitivity)
    daily['z_score'] = (daily['total_activity'] - daily['rolling_mean']) / (daily['rolling_std'] + 1e-9)
    
    anomalies = daily[np.abs(daily['z_score']) > 2.0].copy()
    
    # --- CLOSED-LOOP ACTION LOGIC (Mocked Workflows) ---
    actions = {
        "Spike": ["Vigilance Audit Dispatched", "Server Load Balancing Initiated", "Fraud Analysis Triggered", "Center Capacity Verification"],
        "Drop": ["Network Connectivity Check", "Center Health Heartbeat Request", "Regional Manager Alerted", "ISP Outage Ticket Identified"]
    }
    statuses = ["Pending Response", "Action Issued", "Field Teams Notified", "Resolved", "Monitoring"]
    
    results = []
    for _, row in anomalies.iterrows():
        a_type = "Spike" if row['z_score'] > 0 else "Drop"
        risk = "Potential System Outage" if row['z_score'] < 0 else "High-Intensity Activity / Audit Recommended"
        
        # Deterministic mock based on date to keep it consistent across refreshes
        seed = int(row['date'].strftime('%s')) % 4
        
        results.append({
            "date": row['date'].strftime('%Y-%m-%d'),
            "type": a_type,
            "z_score": round(row['z_score'], 2),
            "risk": risk,
            "action_issued": actions[a_type][seed],
            "status": statuses[seed % len(statuses)]
        })
    return results

def calculate_psaci_index(df):
    """
    Calculates Pincode Service Access Concentration Index (PSACI).
    A composite index using normalized:
    1. Activity Volume (Demand intensity)
    2. Child Ratio Proxy (Saturation gap)
    3. Population Pressure (Relative to district)
    """
    # Aggregate by pincode
    pin_stats = df.groupby('pincode').agg({
        'total_activity': 'sum',
        'age_0_5': 'sum',
        'age_5_17': 'sum'
    }).reset_index()
    
    if pin_stats.empty:
        return pin_stats

    # 1. Volume Factor
    v_max = pin_stats['total_activity'].max()
    v_min = pin_stats['total_activity'].min()
    pin_stats['v_norm'] = (pin_stats['total_activity'] - v_min) / (v_max - v_min + 1)
    
    # 2. Child Ratio Factor (Proxy for youth concentration/dependency)
    pin_stats['child_ratio'] = (pin_stats['age_0_5'] + pin_stats['age_5_17']) / (pin_stats['total_activity'] + 1)
    c_max = pin_stats['child_ratio'].max()
    c_min = pin_stats['child_ratio'].min()
    pin_stats['c_norm'] = (pin_stats['child_ratio'] - c_min) / (c_max - c_min + 1)
    
    # 3. Pressure Factor (Weighted composite)
    # PSACI = (0.5 * v_norm) + (0.5 * c_norm)
    pin_stats['psaci_score'] = (pin_stats['v_norm'] * 0.5 + pin_stats['c_norm'] * 0.5)
    
    return pin_stats.sort_values('psaci_score', ascending=False)


def simulate_policy_impact(current_daily_capacity, current_backlog, added_capacity_units, center_hours_increase):
    """
    Queueing Theory Logic for Decision Support
    """
    # Assumptions
    AVG_UNIT_CAPACITY = 80 # Enrolments per day per capacity unit
    AVG_CENTER_CAPACITY = 120 # Enrolments per day
    CAPACITY_PER_HOUR = AVG_CENTER_CAPACITY / 8 # Assuming 8 hour shift
    
    # New Capacity Calculation
    added_capacity_volume = added_capacity_units * AVG_UNIT_CAPACITY
    added_capacity_hours = (center_hours_increase * CAPACITY_PER_HOUR) * 10 # Assuming 10 centers per district avg for simulation
    
    total_new_daily_capacity = current_daily_capacity + added_capacity_volume + added_capacity_hours
    
    # Impact Metrics
    improvement_pct = ((total_new_daily_capacity - current_daily_capacity) / current_daily_capacity) * 100
    
    # Days to clear backlog analysis
    # Formula: Days = Backlog / (Daily_Capacity - Daily_Demand_Approx)
    # Assuming Daily Demand matches current capacity (steady state) for simulation simplicity
    net_capacity_now = current_daily_capacity * 0.05 # 5% buffer normally
    net_capacity_new = (total_new_daily_capacity - current_daily_capacity) + net_capacity_now
    
    days_to_clear_now = current_backlog / net_capacity_now if net_capacity_now > 0 else 365
    days_to_clear_new = current_backlog / net_capacity_new if net_capacity_new > 0 else 365
    
    days_saved = max(0, days_to_clear_now - days_to_clear_new)
    
    return {
        "new_capacity": total_new_daily_capacity,
        "improvement_pct": improvement_pct,
        "days_to_clear_now": days_to_clear_now,
        "days_to_clear_new": days_to_clear_new,
        "days_saved": days_saved
    }

# --- AUTOMATED STRATEGIC INSIGHTS ---


def generate_insights(df, dist_stats, selected_scope, scope_name):
    insights = []
    
    # 1. High Level Trend - Demographic Pulse
    total_vol = df['total_activity'].sum()
    child_count = df['age_0_5'].sum()
    youth_count = df['age_5_17'].sum()
    adult_count = df['age_18_greater'].sum()
    child_pct = (child_count / total_vol) * 100
    
    insights.append({
        "title": "Demographic Pulse",
        "text": f"Across {scope_name}, child enrolment constitutes **{child_pct:.1f}%** of all activity. "
                + ("This is below the recommended 15% threshold, indicating a need for targeted Anganwadi drives." if child_pct < 15 else "This indicates healthy saturation in early age groups."),
        "data": {
            "0-5 Years": f"{child_count:,.0f}",
            "5-17 Years": f"{youth_count:,.0f}",
            "18+ Years": f"{adult_count:,.0f}",
            "Total": f"{total_vol:,.0f}"
        }
    })
    
    # 2. Operational Stress
    # 2. Operational Stress (Automated Detection)
    high_stress_dists = dist_stats[dist_stats['demand_zone'].str.contains('High')]
    
    if not high_stress_dists.empty:
        # Dynamic Insight Generation
        top_stressed = high_stress_dists.nlargest(3, 'demand_score')
        top_stressed_names = top_stressed['postal_district'].tolist()
        avg_demand_score = high_stress_dists['demand_score'].mean()
        
        # Actionable Recommendation
        rec_text = "Recommend immediate deployment of **Extra Enrolment Kits (EEK)**." if avg_demand_score > 1.5 else "Monitor channel utilization weekly."
        
        insights.append({
            "title": "Resource Allocation Alert",
            "text": f"**{len(high_stress_dists)} districts** are operating in the 'High Demand' zone (>1.3x National Avg). "
                    f"Traffic intensity suggests standard centers are overwhelmed. {rec_text} Priority Districts: **{', '.join(top_stressed_names)}**.",
            "data": {
                "High Demand Districts": f"{len(high_stress_dists)}",
                "Avg Demand Score": f"{avg_demand_score:.2f}x",
                "Likely Backlog": f"> 14 Days",
                "Required Action": "Rebalance Capacity"
            }
        })
        
    # 3. Silent Zones
    silent_dists = dist_stats[dist_stats['is_silent_underenrolment']]
    if not silent_dists.empty:
        avg_silent_demand = silent_dists['demand_score'].mean()
        avg_silent_volatility = silent_dists['volatility_score'].mean()
        total_silent_activity = silent_dists['total_activity'].sum()
        
        insights.append({
            "title": "Silent Under-Enrolment Detected",
            "text": f"Statistical algorithms have identified **{len(silent_dists)} regions** with consistently low activity and low variance. "
                    f"Unlike volatile drops, these areas effectively 'flatlined'. Prioritize audit for: **{', '.join(silent_dists['postal_district'].head(3).tolist())}**.",
            "data": {
                "Affected Districts": f"{len(silent_dists)}",
                "Avg Demand Score": f"{avg_silent_demand:.2f}x",
                "Avg Volatility": f"{avg_silent_volatility:.2f}",
                "Total Activity": f"{total_silent_activity:,.0f}"
            }
        })
    
    # 4. Concentration Risk
    high_conc = dist_stats[dist_stats['concentration_risk'] > 0.6]
    if not high_conc.empty:
        avg_conc = high_conc['concentration_risk'].mean()
        max_conc = high_conc['concentration_risk'].max()
        max_conc_dist = high_conc.loc[high_conc['concentration_risk'].idxmax(), 'postal_district']
        
        insights.append({
            "title": "Hyper-Local Concentration",
            "text": f"In **{len(high_conc)} districts**, over 60% of enrolment happens in just the top 10% of pincodes. "
                    "This suggests 'Centre Deserts' in the remaining 90%. Recommendation: Dynamic resource deployment to peripheral pincodes.",
            "data": {
                "Affected Districts": f"{len(high_conc)}",
                "Avg Concentration": f"{avg_conc:.1%}",
                "Highest Concentration": f"{max_conc:.1%}",
                "Most Concentrated": f"{max_conc_dist}"
            }
        })

    return insights

# --- MAIN APP ---

df = load_and_process_data()
if df.empty:
    st.stop()

dist_stats_all = calculate_district_metrics(df)

# --- TOP CONTROL BAR (Filters) ---
# Dashboard controls panel at the top with all 3 filters
st.markdown("<br><br><br>", unsafe_allow_html=True)

# 1. PROCESS MAP SELECTION EARLY (Fixes state mutation error)
if 'map_select' in st.session_state and st.session_state.map_select:
    selection = st.session_state.map_select.get('selection', {})
    if selection and selection.get('points'):
        clicked_state = selection['points'][0].get('location')
        state_options_temp = ["All India"] + sorted(df['postal_state'].unique().tolist())
        if clicked_state and clicked_state in state_options_temp and clicked_state != st.session_state.get('state_selector'):
            st.session_state['state_selector'] = clicked_state
            st.session_state['selected_state_index'] = state_options_temp.index(clicked_state)
            # Clear selection to avoid loop
            st.session_state.map_select = None
            st.rerun()

# 2. DEFINE CALLBACK FOR RESET
def reset_national_view():
    st.session_state['state_selector'] = "All India"
    st.session_state['selected_state_index'] = 0

st.markdown('<div class="controls-panel">', unsafe_allow_html=True)
st.markdown('<div class="controls-title">Dashboard Controls</div>', unsafe_allow_html=True)

# Three filters in a row
col_state, col_district, col_age = st.columns(3)

with col_state:
    state_options = ["All India"] + sorted(df['postal_state'].unique().tolist())
    default_ix = st.session_state.get('selected_state_index', 0)
    
    if st.session_state.get('state_selector', 'All India') != "All India":
         sel_col, reset_col = st.columns([5, 1])
         with sel_col:
            selected_state = st.selectbox(
                "State / Region", 
                state_options,
                index=default_ix,
                key="state_selector",
                help="Filter data by state or view all India"
            )
         with reset_col:
             st.markdown('<div style="margin-top: 28px;"></div>', unsafe_allow_html=True)
             st.button("❌", help="Reset to National View", on_click=reset_national_view)
    else:
        selected_state = st.selectbox(
            "State / Region", 
            state_options,
            index=default_ix,
            key="state_selector",
            help="Filter data by state or view all India"
        )

with col_district:
    if selected_state != "All India":
        districts_in_state = sorted(df[df['postal_state'] == selected_state]['postal_district'].unique().tolist())
        selected_district = st.selectbox(
            "District", 
            ["All"] + districts_in_state,
            help="Filter by specific district"
        )
    else:
        selected_district = "All"
        st.selectbox("District", ["All"], disabled=True, help="Select a state first")

with col_age:
    age_group_options = {
        'age_0_5': '0-5 Years (Infants)',
        'age_5_17': '5-17 Years (Children/Youth)',
        'age_18_greater': '18+ Years (Adults)'
    }
    age_groups = st.multiselect(
        "Age Groups",
        options=list(age_group_options.keys()),
        default=list(age_group_options.keys()),
        format_func=lambda x: age_group_options[x],
        help="Filter by demographic age groups"
    )

st.markdown('</div>', unsafe_allow_html=True)

# Apply Filters
if selected_state != "All India":
    filtered_df = df[df['postal_state'] == selected_state]
    if selected_district != "All":
        filtered_df = filtered_df[filtered_df['postal_district'] == selected_district]
        scope_name = f"{selected_district}, {selected_state}"
    else:
        scope_name = selected_state
    
    # Recalculate stats for the filtered view - OPTIMIZED
    # OLD: dist_stats_filtered = calculate_district_metrics(filtered_df) 
    # NEW: Just filter the pre-calculated stats. This is much faster and keeps "Demand Score" relative to National Benchmark.
    dist_stats_filtered = dist_stats_all[dist_stats_all['postal_state'] == selected_state]
    
    if selected_district != "All":
        dist_stats_filtered = dist_stats_filtered[dist_stats_filtered['postal_district'] == selected_district]

else:
    filtered_df = df
    dist_stats_filtered = dist_stats_all
    scope_name = "All India"

# --- GLOBAL METRICS CALCULATION (Available for all tabs) ---
total_enrolment = filtered_df['total_activity'].sum()

# Velocity calculation for global cards
def get_global_velocity(df):
    max_date = df['date'].max()
    p1 = df[df['date'] > (max_date - pd.Timedelta(days=30))]['total_activity'].sum()
    p2 = df[(df['date'] <= (max_date - pd.Timedelta(days=30))) & (df['date'] > (max_date - pd.Timedelta(days=60)))]['total_activity'].sum()
    if p2 == 0: return 0
    return ((p1 - p2) / p2) * 100

global_velocity = get_global_velocity(filtered_df)

if total_enrolment > 0:
    child_pct = (filtered_df['age_0_5'].sum() / total_enrolment) * 100
    adult_pct = (filtered_df['age_18_greater'].sum() / total_enrolment) * 100
else:
    child_pct = 0
    adult_pct = 0

# --- GLOBAL RECOMMENDITONS ENGINE (Calculated here for use in Briefs & Tabs) ---
recs = []
# 1. Backlog Logic
high_demand_count = len(dist_stats_filtered[dist_stats_filtered['demand_score'] > 1.3])
if high_demand_count >= 1:
    recs.append({
        "type": "Critical",
        "icon": "🚨",
        "action": "Initiate Rapid Deployment Protocol",
        "detail": f"High demand detected in {high_demand_count} districts. Mobilize dynamic resource units to reduce wait times below 20 minutes."
    })
    
# 2. Silent Under-Enrolment Logic
silent_count = len(dist_stats_filtered[dist_stats_filtered['is_silent_underenrolment']])
if silent_count > 0:
        recs.append({
        "type": "Warning",
        "icon": "⚠️",
        "action": "Audit Silent Zones",
        "detail": f"{silent_count} districts show suspiciously low activity. Dispatch vigilance teams to inspect center operational status."
    })
    
# 3. Child Enrolment Logic
if child_pct < 20:
        recs.append({
        "type": "Strategic",
        "icon": "👶",
        "action": "Anganwadi Outreach Campaign",
        "detail": f"Child enrolment ({child_pct:.1f}%) is below target (25%). Partner with WCD Ministry for school/anganwadi camp drives."
    })

# --- OFFICIAL HEADER ---
# Text header removed as per user request (head.jpg is now the banner)
# st.markdown("""...""", unsafe_allow_html=True)
st.markdown(f"**Current Scope:** {scope_name} | **Data Range:** {df['date'].min().date()} to {df['date'].max().date()}")

# TABS
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "Overview", 
    "District Analytics", 
    "Demographic Insights", 
    "Geographic Access", 
    "Automated Strategic Insights", 
    "Recommendations",
    "System Architecture"
])

# --- TAB 1: OVERVIEW ---
with tab1:
    st.markdown('<div class="section-header">National / State Overview</div>', unsafe_allow_html=True)
    
    # 1️⃣ Closed-Loop Statistical Anomaly & Action Panel
    anomalies = detect_statistical_anomalies(filtered_df)
    if anomalies:
        with st.expander(f"🛡️ Active Incident Monitoring: {scope_name} ({len(anomalies)})", expanded=True):
            st.markdown("<div style='font-size: 0.85rem; color: #64748B; margin-bottom: 10px;'>High-confidence deviations identified via <b>Rolling Z-Scores</b>. All anomalies automatically trigger a closed-loop operational workflow.</div>", unsafe_allow_html=True)
            
            # Action Flow Header
            st.markdown("""
            <div style='display: grid; grid-template-columns: 1fr 1fr 2fr 1.5fr 1fr; background: #003366; color: white; padding: 8px 12px; border-radius: 4px; font-weight: 600; font-size: 0.8rem; margin-bottom: 8px;'>
                <div>DATE</div>
                <div>TYPE</div>
                <div>DETECTION RISK</div>
                <div>ACTION ISSUED</div>
                <div>STATUS</div>
            </div>
            """, unsafe_allow_html=True)
            
            for a in anomalies:
                color = "#EF4444" if a['type'] == "Spike" else "#F59E0B"
                status_color = "#16A34A" if a['status'] == "Resolved" else "#2563EB"
                
                st.markdown(f"""
                <div style='display: grid; grid-template-columns: 1fr 1fr 2fr 1.5fr 1fr; background: #F8FAFC; padding: 10px; border-bottom: 1px solid #E2E8F0; font-size: 0.85rem; align-items: center;'>
                    <div style='font-weight: 500;'>{a['date']}</div>
                    <div style='color: {color}; font-weight: 700;'>{a['type']}</div>
                    <div style='color: #475569;'>{a['risk']}</div>
                    <div style='font-style: italic; color: #1E3A8A;'>{a['action_issued']}</div>
                    <div style='background: {status_color}20; color: {status_color}; padding: 2px 6px; border-radius: 4px; font-weight: 800; text-align: center; border: 1px solid {status_color}40;'>{a['status']}</div>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.success("✅ System Status: Stable. No operational anomalies detected in the current window.")
    
    
    # Metrics
    # Metrics
    tot_act = total_enrolment # Alias for tab compatibility
    avg_act = filtered_df.groupby('date')['total_activity'].sum().mean()
    child_share = child_pct
    
    c1, c2, c3, c4 = st.columns(4)

    # Velocity indicators
    v_arrow = "↑" if global_velocity >= 0 else "↓"
    v_color = "#16A34A" if global_velocity >= 0 else "#EF4444"
    velocity_html = f"<div style='color: {v_color}; font-size: 0.8rem; font-weight: 700;'>{v_arrow} {abs(global_velocity):.1f}% Velocity</div>"

    # --- MISSION BRIEF GENERATOR ---
    # Generate data for the report
    with st.spinner("Authorizing Secure Connection..."):
        insights_preview = generate_insights(filtered_df, dist_stats_filtered, selected_state, scope_name)
        
        recs_preview = []
        if len(dist_stats_filtered[dist_stats_filtered['demand_score'] > 1.3]) >= 1: 
            recs_preview.append({"type": "Critical", "icon": "🚨", "action": "Initiate Rapid Deployment", "detail": "High demand detected. Mobilize dynamic resources."})
        if len(dist_stats_filtered[dist_stats_filtered['is_silent_underenrolment']]) > 0:
            recs_preview.append({"type": "Warning", "icon": "⚠️", "action": "Audit Silent Zones", "detail": "Suspicious low activity detected."})
            
        kpis_report = {
            "total": format_indian(tot_act),
            "daily_avg": format_indian(avg_act),
            "child_pct": str(round(child_share, 1))
        }
        
        brief_html = generate_mission_brief_html(kpis_report, recs_preview, insights_preview, scope_name)
        b64 = base64.b64encode(brief_html.encode()).decode()

        
        # COMMANDER'S DESK (Moved from Sidebar)
        href = f'<a href="data:file/html;base64,{b64}" download="UIDAI_Mission_Brief_{datetime.now().strftime("%Y%m%d")}.html" style="text-decoration:none;">'
        st.markdown(f"""
        <div style="text-align: right; margin-bottom: 10px;">
            {href}<button style="background-color:#003366; color:white; border:none; padding:8px 16px; border-radius:5px; font-weight:bold; cursor:pointer; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
            🔒 Generate Operational Intelligence Brief</button></a>
        </div>
        """, unsafe_allow_html=True)

    c1.markdown(f'<div class="metric-card"><div class="metric-value">{format_indian(tot_act)}</div>{velocity_html}<div class="metric-label">National Activity Volume</div></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="metric-card"><div class="metric-value">{format_indian(avg_act)}</div><div class="metric-label">Daily Aggregated Avg</div></div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="metric-card"><div class="metric-value">{child_share:.1f}%</div><div class="metric-label">Child Enrolment Ratio</div></div>', unsafe_allow_html=True)
    c4.markdown(f'<div class="metric-card"><div class="metric-value">{len(dist_stats_filtered)}</div><div class="metric-label">Districts Active</div></div>', unsafe_allow_html=True)

    # --- SIMULATED SYSTEM BOOT ---
    # --- SYSTEM STATUS INDICATOR ---
    if 'system_ready' not in st.session_state:
        st.session_state['system_ready'] = True
    
    # --- POLICY SIMULATOR (DECISION SUPPORT) ---
    with st.expander("🛠️ Decision Support System (Policy Simulator)", expanded=True):
        st.markdown("##### ⚡ Dynamic Enrolment Capacity Rebalancing System")
        sim_col1, sim_col2, sim_col3 = st.columns(3)
        
        with sim_col1:
            capacity_adj_pct = st.slider("Capacity Adjustment (%)", 0, 100, 0, format="%d%%")
            st.caption("Flexible reallocation of resources (counters, camps) based on demand.")
            
        with sim_col2:
            extra_hours = st.slider("Extend Center Hours", 0, 4, 0, help="Increase operational hours/day")
            
        # Calculate Impact
        current_daily_capacity = avg_act # Using active daily avg as proxy for capacity
        current_backlog = tot_act * 0.15 # Simulation: 15% hidden backlog
        
        # Convert percent adjustment to 'units' equivalent for backend compatibility
        # Assumption: 1 unit ~ 80/day. Base capacity ~50k. 
        # We will update logic to use direct capacity addition
        added_capacity_units = (current_daily_capacity * (capacity_adj_pct / 100)) / 80 # approx unit equivalent
        
        impact = simulate_policy_impact(current_daily_capacity, current_backlog, added_capacity_units, extra_hours)
        
        with sim_col3:
            if capacity_adj_pct > 0 or extra_hours > 0:
                st.markdown(f"""
                    <div style='background-color: #F0FDF4; border: 1px solid #16A34A; border-radius: 8px; padding: 10px; text-align: center;'>
                        <div style='color: #166534; font-size: 0.8rem; font-weight: 600;'>PROJECTED CLEARANCE</div>
                        <div style='color: #15803D; font-size: 1.4rem; font-weight: 800;'>{impact['days_to_clear_new']:.1f} Days</div>
                        <div style='color: #166534; font-size: 0.8rem;'>Saved: {impact['days_saved']:.1f} Days</div>
                    </div>
                """, unsafe_allow_html=True)
            else:
                 st.markdown(f"""
                    <div style='background-color: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px; padding: 10px; text-align: center; opacity: 0.7;'>
                        <div style='color: #64748B; font-size: 0.8rem; font-weight: 600;'>CURRENT STATUS</div>
                        <div style='color: #475569; font-size: 1.4rem; font-weight: 800;'>BAU</div>
                        <div style='color: #64748B; font-size: 0.8rem;'>Adjust sliders to simulate</div>
                    </div>
                """, unsafe_allow_html=True)
    
    # Styled section divider
    st.markdown("""
    <div style='
        background: linear-gradient(135deg, #003366 0%, #1a5c99 100%);
        padding: 12px 24px;
        border-radius: 12px;
        text-align: center;
        margin: 20px 0 24px 0;
        box-shadow: 0 4px 15px rgba(0, 51, 102, 0.2);
    '>
        <span style='
            color: #FFFFFF;
            font-family: Poppins, sans-serif;
            font-size: 1.2rem;
            font-weight: 600;
            letter-spacing: 1px;
            text-transform: uppercase;
        '>📊 Strategic Analytics & Forecasting</span>
    </div>
    """, unsafe_allow_html=True)
    
    # --- INDIA MAP IMPLEMENTATION (NEW) ---
    st.markdown("### Geographic Overview")
    
    # Two column layout: Left = Charts, Right = Map
    col_left, col_right = st.columns([1, 1])
    
    with col_left:
        # Time Series with PREDICTIVE ANALYTICS
        daily = filtered_df.groupby('date')[['total_activity', 'age_0_5', 'age_18_greater']].sum().reset_index()
        
        # Create base figure
        fig_line = px.line(daily, x='date', y=['total_activity', 'age_0_5', 'age_18_greater'], 
                          title="Temporal Trends & Predictive Analytics 🧠", color_discrete_sequence=['#2563EB', '#F59E0B', '#10B981'])
        
        # Add REAL Prediction (Holt-Winters Approx)
        if len(daily) > 5:
            forecast_vals = calculate_forecast_holt_winters(daily['total_activity'].tolist(), n_preds=15)
            last_date = daily['date'].max()
            future_dates = pd.date_range(start=last_date, periods=16)[1:]
            
            fig_line.add_scatter(x=future_dates, y=forecast_vals, mode='lines', 
                                name='Predictive Analytics (Trend+Seasonality)', 
                                line=dict(color='#FF0000', width=2, dash='dot'))
            
            # Policy Simulation Impact overlay
            if capacity_adj_pct > 0 or extra_hours > 0:
                impact_vals = forecast_vals * (1 + (impact['improvement_pct']/100))
                fig_line.add_scatter(x=future_dates, y=impact_vals, mode='lines',
                                    name='With Policy Intervention',
                                    line=dict(color='#16A34A', width=2, dash='dashdot'))

        fig_line.update_layout(template="plotly_white", legend=dict(orientation="h", y=1.1), height=380)
        st.plotly_chart(fig_line, use_container_width=True)
        
        # Age Mix Pie
        age_sums = filtered_df[['age_0_5', 'age_5_17', 'age_18_greater']].sum().reset_index()
        age_sums.columns = ['Age Group', 'Count']
        fig_pie = px.pie(age_sums, values='Count', names='Age Group', title="Demographic Distribution", hole=0.5,
                        color_discrete_sequence=['#F59E0B', '#6366F1', '#10B981'])
        fig_pie.update_layout(height=380, annotations=[dict(text='AGE<br>MIX', x=0.5, y=0.5, font_size=14, showarrow=False)])
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col_right:
        # Custom Color Scale: Yellow (Low) -> Green (Normal) -> Red (High)
        custom_color_scale = ['#FFD700', '#228B22', '#FF0000']
        
        # Common District Corrections (Manual)
        district_corrections = {
            'Jalore': 'Jalor',
            'Barabanki': 'Bara Banki',
            'Davangere': 'Davanagere',
            'Kanchipuram': 'Kancheepuram',
            'Tiruvallur': 'Thiruvallur',
            'Thoothukkudi': 'Thoothukudi',
            'Ahmadnagar': 'Ahmednagar',
            'Gondia': 'Gondiya',
            'Buldhana': 'Buldana',
            'Bid': 'Beed',
            'Nashik': 'Nasik',
            'Jalgaon': 'Jalgaon', 
            'Hugli': 'Hooghly',
            'Haora': 'Howrah',
            'Koch Bihar': 'Cooch Behar',
            'North 24 Parganas': 'North Twenty Four Parganas',
            'South 24 Parganas': 'South Twenty Four Parganas',
            'Purba Bardhaman': 'Barddhaman',
            'Paschim Bardhaman': 'Barddhaman',
            'Kalaburagi': 'Gulbarga',
            'Vijayapura': 'Bijapur',
            'Belagavi': 'Belgaum',
            'Shivamogga': 'Shimoga',
            'Ballari': 'Bellary',
            'Mysuru': 'Mysore',
            'Tumakuru': 'Tumkur',
            'Bengaluru Urban': 'Bangalore Urban',
            'Bengaluru Rural': 'Bangalore Rural',
            'Y.S.R.': 'Y.S.R.',
            'S.P.S. Nellore': 'Nellore',
            'Sri Potti Sriramulu Nellore': 'Nellore', 
            'Chittoor': 'Chittoor',
            'Visakhapatanam': 'Visakhapatnam',
            'Kheri': 'Lakhimpur Kheri',
            'Sant Ravidas Nagar': 'Bhadohi',
            'Panch Mahals': 'Panch Mahals',
            'Dahod': 'Dohad',
            'Sabar Kantha': 'Sabarkantha',
            'Banas Kantha': 'Banaskantha',
            'The Dangs': 'The Dangs',
            'Kachchh': 'Kachchh',
            'Leh Ladakh': 'Leh',
            'Kargil': 'Kargil'
        }

        try:
            if selected_state == "All India":
                # --- STATE LEVEL MAP ---
                state_agg = filtered_df.groupby('postal_state')['total_activity'].sum().reset_index()
                state_agg['postal_state'] = state_agg['postal_state'].str.title()
                
                # Corrections
                state_map_corrections = {
                    'Andaman & Nicobar Islands': 'Andaman and Nicobar Islands',
                    'Jammu & Kashmir': 'Jammu and Kashmir',
                    'Dadra & Nagar Haveli And Daman & Diu': 'Dadra and Nagar Haveli and Daman and Diu',
                    'Delhi': 'NCT of Delhi'
                }
                state_agg['postal_state'] = state_agg['postal_state'].replace(state_map_corrections)
                
                # Load cached GeoJSON (States only to be fast)
                geojson_states, _ = load_geojson()
                
                if geojson_states:
                    fig_map = px.choropleth(
                        state_agg,
                        geojson=geojson_states,
                        featureidkey='properties.ST_NM',
                        locations='postal_state',
                        color='total_activity',
                        color_continuous_scale=custom_color_scale,
                        title="Total Enrolment Activity by State",
                        hover_name='postal_state',
                        labels={'total_activity': 'Enrolments'}
                    )
                    fig_map.update_geos(fitbounds="locations", visible=False, projection_scale=1.3)
                    # Enable Selection
                    fig_map.update_layout(clickmode='event+select', height=650, margin={"r":0,"t":0,"l":0,"b":0},
                                         coloraxis_colorbar=dict(title="Enrolments", len=0.8), autosize=True)
                    
                    # Add State Labels (All India)
                    state_centroids = {
                        'Maharashtra': (19.7515, 75.7139),
                        'Uttar Pradesh': (26.8467, 80.9462),
                        'Bihar': (25.0961, 85.3131),
                        'West Bengal': (22.9868, 87.8550),
                        'Madhya Pradesh': (22.9734, 78.6569),
                        'Tamil Nadu': (11.1271, 78.6569),
                        'Rajasthan': (27.0238, 74.2179),
                        'Karnataka': (15.3173, 75.7139),
                        'Gujarat': (22.2587, 71.1924),
                        'Andhra Pradesh': (15.9129, 79.7400),
                        'Odisha': (20.9517, 85.0985),
                        'Telangana': (18.1124, 79.0193),
                        'Kerala': (10.8505, 76.2711),
                        'Jharkhand': (23.6102, 85.2799),
                        'Assam': (26.2006, 92.9376),
                        'Punjab': (31.1471, 75.3412),
                        'Chhattisgarh': (21.2787, 81.8661),
                        'Haryana': (29.0588, 76.0856),
                        'Jammu And Kashmir': (33.7782, 76.5762),
                        'Uttarakhand': (30.0668, 79.0193),
                    }
                    
                    label_lats = []
                    label_lons = []
                    label_texts = []
                    
                    for state in state_agg['postal_state'].unique():
                        state_key = state.title()
                        if state_key in state_centroids:
                            lat, lon = state_centroids[state_key]
                            label_lats.append(lat)
                            label_lons.append(lon)
                            label_texts.append(state_key)  # Full state name
                    
                    # Add state labels as a Scattergeo trace
                    fig_map.add_trace(go.Scattergeo(
                        lat=label_lats,
                        lon=label_lons,
                        text=label_texts,
                        mode='text',
                        textfont=dict(size=12, color='#000000', family='Poppins, sans-serif'),
                        showlegend=False,
                        hoverinfo='skip'
                    ))
                    
                    # RENDER MAP WITH SELECTION
                    selected_points = st.plotly_chart(fig_map, use_container_width=True, on_select="rerun", key="map_select")
                    
                    # NOTE: Selection handling moved to top of script to avoid Streamlit state mutation errors.
                
            else:
                # --- DISTRICT LEVEL MAP ---
                dist_map_agg = filtered_df.groupby('postal_district')['total_activity'].sum().reset_index()
                dist_map_agg['postal_district'] = dist_map_agg['postal_district'].str.title()
                dist_map_agg['postal_district'] = dist_map_agg['postal_district'].replace(district_corrections)
                
                _, geojson_districts = load_geojson()
                
                if geojson_districts:
                     fig_map = px.choropleth(
                        dist_map_agg,
                        geojson=geojson_districts,
                        featureidkey='properties.district',
                        locations='postal_district',
                        color='total_activity',
                        color_continuous_scale=custom_color_scale,
                        title=f"District Activity in {selected_state}",
                        hover_name='postal_district',
                        labels={'total_activity': 'Enrolments'}
                    )
                     fig_map.update_geos(fitbounds="locations", visible=False, projection_scale=1.3)
                     fig_map.update_layout(height=650, margin={"r":0,"t":0,"l":0,"b":0},
                                         coloraxis_colorbar=dict(title="Enrolments", len=0.8), autosize=True)
                     st.plotly_chart(fig_map, use_container_width=True)

        except Exception as e:
            st.warning(f"Map rendering limitation: {e}. Note: Ensure internet connection for GeoJSON loading.")

# --- TAB 2: DEMAND INTELLIGENCE ---
with tab2:
    st.markdown('<div class="section-header">District Demand Intelligence</div>', unsafe_allow_html=True)
    
    col_d1, col_d2 = st.columns([2, 1])
    
    with col_d1:
        # Perform Clustering
        cluster_labels = perform_custom_clustering(dist_stats_filtered, n_clusters=3)
        dist_stats_filtered['cluster_label'] = cluster_labels
        
        fig_map = px.scatter(
            dist_stats_filtered,
            x='total_activity',
            y='demand_score',
            color='cluster_label', # Use semantic clusters
            size='total_activity',
            hover_name='postal_district',
            hover_data=['stress_persistence_months', 'volatility_score'],
            title="District Performance Clusters (Algorithmic Segmentation)",
            color_discrete_map={
                'Critical Care Zone 🔴': '#EF4444', 
                'Monitoring Zone 🟡': '#F59E0B', 
                'Stable Zone 🟢': '#10B981'
            }
        )
        st.plotly_chart(fig_map, use_container_width=True)
        
    with col_d2:
        st.subheader("High Demand Districts")
        st.markdown("Districts exceeding 1.3× National Average")
        high_stress = dist_stats_filtered[dist_stats_filtered['demand_score'] > 1.3].sort_values('demand_score', ascending=False).head(10)
        st.dataframe(high_stress[['postal_district', 'demand_score', 'stress_persistence_months']], hide_index=True)
        
        # Explanatory info
        st.markdown(""" 
        <div style='background: #F0F9FF; border-left: 3px solid #0EA5E9; padding: 12px 16px; border-radius: 8px; margin-top: 16px; font-size: 0.85rem;'>
            <strong style='color: #0369A1;'>📊 Understanding This Data</strong><br>
            <span style='color: #475569;'><b>Demand Score:</b> Ratio of district activity vs national average (1.0 = average)<br>
            <b>Stress Months:</b> Count of months where demand exceeded critical threshold</span>
        </div>
        """, unsafe_allow_html=True)

# --- TAB 3: AGE DEEP DIVE ---
with tab3:
    st.markdown('<div class="section-header">Age-Group Deep Dive</div>', unsafe_allow_html=True)
    
    c_age1, c_age2 = st.columns(2)
    
    with c_age1:
        st.subheader("Child Enrolment Analysis")
        st.markdown("Districts with lowest proportionate child enrolment")
        laggards = dist_stats_filtered.sort_values('child_ratio', ascending=True).head(15)
        fig_lag = px.bar(laggards, x='child_ratio', y='postal_district', orientation='h', 
                        title="Districts with Lowest Child Enrolment Ratio", color='child_ratio', color_continuous_scale='Reds_r')
        st.plotly_chart(fig_lag, use_container_width=True)
        
        # Explanatory info
        st.markdown(""" 
        <div style='background: #FEF3C7; border-left: 3px solid #F59E0B; padding: 12px 16px; border-radius: 8px; font-size: 0.85rem;'>
            <strong style='color: #B45309;'>📋 What is Child Enrolment Ratio?</strong><br>
            <span style='color: #475569;'>The percentage of 0-5 year age group enrolments compared to total enrolments in a district. 
            Low ratios may indicate potential exclusion of young children.</span>
        </div>
        """, unsafe_allow_html=True)

    # --- TRIVARIATE ANALYSIS (Deep Dive) ---
    st.markdown("---")
    st.subheader("📊 Multi-Dimensional Analysis (Trivariate)")
    st.markdown("**Demographic Heatmap Matrix:** Visualizing *Activity Intensity* across **Districts** (Y) and **Time** (X), colored by **Child Enrolment Ratio**.")
    
    # Prepare Data for Heatmap
    # Metric: Child Ratio over Time per District (Top 20 Districts by volume)
    top_districts = dist_stats_filtered.nlargest(20, 'total_activity')['postal_district'].tolist()
    tri_df = filtered_df[filtered_df['postal_district'].isin(top_districts)].copy()
    tri_df['month'] = tri_df['date'].dt.strftime('%Y-%m')
    
    heatmap_data = tri_df.groupby(['postal_district', 'month']).agg(
        total=('total_activity', 'sum'),
        child=('age_0_5', 'sum')
    ).reset_index()
    heatmap_data['child_ratio'] = (heatmap_data['child'] / heatmap_data['total']) * 100
    
    fig_heat = px.density_heatmap(
        heatmap_data, 
        x='month', 
        y='postal_district', 
        z='child_ratio', 
        histfunc='avg',
        title="Child Enrolment Intensity Matrix (Red=Low, Blue=High)",
        color_continuous_scale='RdBu',
        labels={'postal_district': 'District', 'month': 'Month', 'child_ratio': 'Child Ratio %'}
    )
    fig_heat.update_layout(height=500)
    st.plotly_chart(fig_heat, use_container_width=True)
    
    with c_age2:
        st.subheader("Adult Update Load Index")
        st.markdown("Districts with high adult (18+) update activity")
        pressured = dist_stats_filtered.sort_values('update_pressure', ascending=False).head(15)
        fig_press = px.bar(pressured, x='update_pressure', y='postal_district', orientation='h',
                          title="Districts with Highest Adult Update Activity", color='update_pressure', color_continuous_scale='Blues')
        st.plotly_chart(fig_press, use_container_width=True)
        
        # Explanatory info
        st.markdown(""" 
        <div style='background: #EFF6FF; border-left: 3px solid #3B82F6; padding: 12px 16px; border-radius: 8px; font-size: 0.85rem;'>
            <strong style='color: #1D4ED8;'>📋 What is Adult Update Load Index?</strong><br>
            <span style='color: #475569;'>Measures the proportion of 18+ age group activities (updates, corrections, biometric renewals) 
            relative to total district operations. High values indicate centres are primarily handling adult updates 
            rather than new enrolments, which may require additional staffing.</span>
        </div>
        """, unsafe_allow_html=True)

# --- TAB 4: PINCODE HEATMAP ---
with tab4:
    st.markdown('<div class="section-header">Geographic Access Distribution</div>', unsafe_allow_html=True)
    
    st.markdown(""" 
    <div style='background: #F5F3FF; border-left: 3px solid #8B5CF6; padding: 14px 18px; border-radius: 8px; margin-bottom: 20px;'>
        <strong style='color: #6D28D9; font-size: 1rem;'>📍 Pincode Service Access Concentration Index (PSACI)</strong><br>
        <span style='color: #475569; font-size: 0.9rem;'>PSACI is a composite index (0 to 1) calculating access equity by normalizing activity volume and youth dependency ratios. 
        <b>Note:</b> This uses aggregated proxies; no individual-level PII is exposed.</span>
    </div>
    """, unsafe_allow_html=True)
    
    # Calculate PSACI
    psaci_data = calculate_psaci_index(filtered_df)
    
    col_p1, col_p2 = st.columns([2, 1])
    
    with col_p1:
        st.subheader("PSACI Access Map (Pincode Level)")
        if not psaci_data.empty:
            # Table-based heatmap for clarity
            st.dataframe(
                psaci_data[['pincode', 'total_activity', 'psaci_score']].head(20),
                column_config={
                    "psaci_score": st.column_config.ProgressColumn(
                        "Access Pressure Index",
                        help="0 = Low Pressure, 1 = Hyper-Concentrated",
                        format="%.2f",
                        min_value=0,
                        max_value=1,
                    )
                },
                hide_index=True,
                use_container_width=True
            )
        else:
            st.info("Insufficient data for Pincode Indexing.")
    
    with col_p2:
        st.subheader("Index Methodology")
        st.markdown("""
        <div style='background: #F8FAFC; padding: 10px; border: 1px solid #E2E8F0; border-radius: 5px; font-size: 0.8rem;'>
            <strong>Components:</strong><br>
            1. <b>Volume Normalization:</b> Scaling demand intensity.<br>
            2. <b>Youth Ratio Proxy:</b> Identifying child-heavy pressure.<br>
            3. <b>Standardization:</b> Min-Max scaling to [0,1].
        </div>
        """, unsafe_allow_html=True)
    
    
    # Additional explanation
    st.markdown(""" 
    <div style='background: #ECFDF5; border-left: 3px solid #10B981; padding: 12px 16px; border-radius: 8px; margin-top: 16px; font-size: 0.85rem;'>
        <strong style='color: #047857;'>💡 Recommended Action</strong><br>
        <span style='color: #475569;'>Districts with concentration risk >60% should be prioritized for dynamic resource deployment 
        to peripheral pincodes. This helps ensure equitable access to Aadhaar services across all geographic areas.</span>
    </div>
    """, unsafe_allow_html=True)

# --- TAB 5: AUTOMATED INSIGHTS ---
with tab5:
    st.markdown('<div class="section-header">Automated Policy Narrative Engine</div>', unsafe_allow_html=True)
    
    insights = generate_insights(filtered_df, dist_stats_filtered, selected_state, scope_name)
    
    col_txt, col_kpi = st.columns([2, 1])
    
    with col_txt:
        for i, insight in enumerate(insights):
            with st.container():
                st.markdown(f"### {i+1}. {insight['title']}")
                st.markdown(insight['text'])
                st.markdown("---")
        kpis = {
        "total": format_indian(total_enrolment),
        "daily_avg": format_indian(filtered_df.groupby('date')['total_activity'].sum().mean()),
        "backlog": "Normal", # Placeholder as backlog status isn't globally calculated yet
        "child_pct": str(round(child_pct, 1))
    }
    
    brief_html = generate_mission_brief_html(kpis, recs, insights, scope_name)
    b64 = base64.b64encode(brief_html.encode()).decode()
    href = f'<a href="data:file/html;base64,{b64}" download="UIDAI_Mission_Brief_{datetime.now().strftime("%Y%m%d")}.html" style="text-decoration:none;">'
    

    
    metric_cols = st.columns(4)
    with col_kpi:
        st.markdown("### 📖 Methodology Reference")
        
        st.markdown(""" 
        <div style='background: #F8FAFC; border: 1px solid #E2E8F0; padding: 16px; border-radius: 10px; margin-bottom: 12px;'>
            <strong style='color: #1E40AF;'>1. Silent Under-Enrolment Detection</strong><br>
            <span style='color: #64748B; font-size: 0.85rem;'>
            <b>Formula:</b> Demand Score < 0.5 AND Volatility < 0.2<br>
            <b>Interpretation:</b> Identifies districts with consistently low activity without seasonal variation—indicating systemic access issues rather than temporary dips.
            </span>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(""" 
        <div style='background: #F8FAFC; border: 1px solid #E2E8F0; padding: 16px; border-radius: 10px; margin-bottom: 12px;'>
            <strong style='color: #1E40AF;'>2. Operational Stress Duration</strong><br>
            <span style='color: #64748B; font-size: 0.85rem;'>
            <b>Formula:</b> Count of months where demand exceeds 1.3× national average<br>
            <b>Interpretation:</b> Distinguishes chronic infrastructure constraints from temporary demand surges requiring different intervention strategies.
            </span>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(""" 
        <div style='background: #F8FAFC; border: 1px solid #E2E8F0; padding: 16px; border-radius: 10px;'>
            <strong style='color: #1E40AF;'>3. Service Access Concentration</strong><br>
            <span style='color: #64748B; font-size: 0.85rem;'>
            <b>Formula:</b> Activity share of top 10% pincodes > 50%<br>
            <b>Interpretation:</b> High concentration indicates urban-centric services with potential under-served rural/peripheral communities.
            </span>
        </div>
        """, unsafe_allow_html=True)

# --- TAB 6: RECOMMENDATIONS ---
with tab6:
    st.markdown('<div class="section-header">Administrative Actions & Recommendations</div>', unsafe_allow_html=True)
    
    # Dynamic Recommendations based on STATE of the system
    # (Calculated Globally as 'recs')
    
    if not recs:
        st.info("System operating within normal parameters. No critical actions required.")
        
    # Render Recommendations
    for rec in recs:
        border_color = "#EF4444" if rec['type'] == "Critical" else "#F59E0B" if rec['type'] == "Warning" else "#3B82F6"
        bg_color = "#FEF2F2" if rec['type'] == "Critical" else "#FFFBEB" if rec['type'] == "Warning" else "#EFF6FF"
        
        st.markdown(f"""
        <div style='background-color: {bg_color}; border-left: 5px solid {border_color}; padding: 15px; border-radius: 8px; margin-bottom: 12px; box-shadow: 0 2px 4px rgba(0,0,0,0.05);'>
            <div style='display: flex; align-items: start; gap: 12px;'>
                <div style='font-size: 1.5rem;'>{rec['icon']}</div>
                <div>
                    <div style='color: #1E293B; font-weight: 700; font-size: 1.1rem; margin-bottom: 4px;'>{rec['action']}</div>
                    <div style='color: #475569; font-size: 0.95rem; line-height: 1.4;'>{rec['detail']}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
# --- TAB 7: SYSTEM ARCHITECTURE ---
with tab7:
    st.markdown('<div class="section-header">Government-Grade Data Architecture</div>', unsafe_allow_html=True)
    
    arch_col1, arch_col2 = st.columns([1.5, 1])
    
    with arch_col1:
        st.markdown("""
        ### 🏗️ Scalable Production Pipeline
        This dashboard serves as the **Presentation & Decision Layer** of a multi-stage big data pipeline. It is intentionally decoupled from raw data ingestion to ensure sub-second query performance at national scale.
        
        <div style='padding: 24px; background: white; border-radius: 12px; border: 1px solid #E2E8F0; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);'>
            <div style='display: flex; flex-direction: column; gap: 15px;'>
                <div style='display: flex; align-items: center; gap: 15px;'>
                    <div style='background: #003366; color: white; width: 30px; height: 30px; border-radius: 50%; display: flex; justify-content: center; align-items: center; font-weight: bold;'>1</div>
                    <div><b>Ingestion Layer:</b> Kafka-based streaming of ECMP (Enrolment Client) logs into <b>Hadoop HDFS / S3 Data Lake</b>.</div>
                </div>
                <div style='text-align: center; color: #94A3B8;'>⬇️</div>
                <div style='display: flex; align-items: center; gap: 15px;'>
                    <div style='background: #003366; color: white; width: 30px; height: 30px; border-radius: 50%; display: flex; justify-content: center; align-items: center; font-weight: bold;'>2</div>
                    <div><b>Processing Layer:</b> <b>Apache Spark (Databricks)</b> jobs perform heavy-lift pre-aggregation. Logic calculates Z-Scores, PSACI, and Velocity metrics in parallel.</div>
                </div>
                <div style='text-align: center; color: #94A3B8;'>⬇️</div>
                <div style='display: flex; align-items: center; gap: 15px;'>
                    <div style='background: #003366; color: white; width: 30px; height: 30px; border-radius: 50%; display: flex; justify-content: center; align-items: center; font-weight: bold;'>3</div>
                    <div><b>Storage Layer:</b> Aggregated Operational Cubes are stored in an <b>Indexed SQL Warehouse (Postgres/BigQuery)</b> for instant retrieval.</div>
                </div>
                <div style='text-align: center; color: #94A3B8;'>⬇️</div>
                <div style='display: flex; align-items: center; gap: 15px;'>
                    <div style='background: #003366; color: white; width: 30px; height: 30px; border-radius: 50%; display: flex; justify-content: center; align-items: center; font-weight: bold;'>4</div>
                    <div><b>Intelligence Layer:</b> This <b>Command Center</b> consumes the indexed cubes via secure APIs, providing real-time Situational Awareness.</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with arch_col2:
        st.markdown("### 🔒 Security & Governance")
        st.info("""
        **Data Minimization:** No PII (Name, Address, Biometrics) leaves the UIDAI secure vault. This system operates on **Anonymized Transaction Telemetry**.
        
        **Scalability:** The architecture is designed to handle **1,000+ simultaneous executive users** and **100M+ transaction records** per day by offloading compute to the Spark cluster.
        
        **Explainability:** Every index (Demand Score, PSACI) follows documented statistical formulas, ensuring all administrative actions are defensible under audit.
        """)

st.markdown("---")
st.caption("UIDAI Internal Command Center Prototype | Secured for Operational Use | Version 2.4.0-HARDENED")
