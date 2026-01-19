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
from fpdf import FPDF
import calendar
from backend.statistical_engine import (
    calculate_district_metrics,
    perform_custom_clustering,
    calculate_forecast_holt_winters,
    detect_statistical_anomalies,
    calculate_psaci_index,
    simulate_policy_impact,
    calculate_campaign_impact_v2
)

# --- HEADER ---
header_html = """
    <div class="fixed-header">
        <div class="header-content">
            <div class="header-left">
                <span class="header-brand">UIDAI Analytics</span>
            </div>
            <div class="header-right">
            </div>
        </div>
    </div>
"""

# --- CONFIGURATION & AESTHETICS ---
st.set_page_config(page_title="UIDAI Analytics Command Center", layout="wide", page_icon="G")

st.markdown(f"""
    <style>
    /* Fixed Header encompassing full width */
    .fixed-header {{
        position: fixed;
        top: 0;
        left: 0;
        width: 100vw;
        height: 60px;
        background-color: #003366;
        border-bottom: 3px solid #FF9933; /* Saffron border */
        z-index: 9999999;
        display: flex;
        align-items: center;
        padding: 0 40px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.2);
    }}
    .header-content {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        width: 100%;
        max-width: 1400px;
        margin: 0 auto;
    }}
    .header-left {{
        display: flex;
        align-items: center;
        gap: 15px;
    }}
    .header-brand {{
        color: white;
        font-weight: 700;
        font-size: 1.25rem;
        letter-spacing: 0.5px;
    }}
    .header-title {{
        color: white;
        font-weight: 400;
        font-size: 1.1rem;
        border-left: 1px solid rgba(255,255,255,0.3);
        padding-left: 15px;
        letter-spacing: 0.5px;
    }}
    .how-it-works {{
        color: #003366 !important;
        text-decoration: none !important;
        font-size: 0.85rem;
        font-weight: 700;
        padding: 6px 16px;
        background-color: #FFFFFF !important;
        border-radius: 6px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        display: inline-block;
    }}
    .how-it-works:hover {{
        background-color: #F8FAFC !important;
        color: #2563EB !important;
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
    }}
    
    /* Hide all Streamlit-specific elements for a custom portal look */
    header {{visibility: hidden !important;}}
    footer {{visibility: hidden !important;}}
    #MainMenu {{visibility: hidden !important;}}
    [data-testid="stDecoration"] {{display: none !important;}}
    [data-testid="stHeader"] {{display: none !important;}}
    [data-testid="stStatusWidget"] {{display: none !important;}}
    
    /* Push content down to account for fixed header */
    [data-testid="stSidebar"] {{
        margin-top: 60px;
        height: calc(100vh - 60px);
        background-color: #FDFDFD;
        border-right: 1px solid #E2E8F0;
    }}

    .block-container {{
        padding-top: 80px !important;
    }}

    /* --- PREMIUM GOVERNMENT THEME --- */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        letter-spacing: -0.01em;
    }}
    
    h1, h2, h3, h4, h5, h6, .section-header {{
        font-family: 'Poppins', 'Inter', sans-serif;
        font-weight: 600;
    }}

    .stApp {{
        background: linear-gradient(135deg, #F8FAFC 0%, #EEF2F7 100%);
    }}

    /* Metrics Cards */
    .metric-card {{
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
    }}
    .metric-card:hover {{
        transform: translateY(-5px);
        box-shadow: 0 12px 30px rgba(0, 51, 102, 0.15);
    }}
    .metric-value {{
        color: #003366;
        font-weight: 800;
        font-size: 2.2rem;
        line-height: 1.2;
        font-family: 'Poppins', sans-serif;
        background: linear-gradient(135deg, #003366 0%, #1a5c99 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }}
    .metric-label {{
        color: #64748B;
        font-size: 0.85rem;
        margin-top: 8px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
    }}
    
    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 10px;
        background-color: #ffffff;
        padding: 10px 10px 0px 10px;
        border-radius: 10px 10px 0 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }}
    .stTabs [data-baseweb="tab"] {{
        background-color: transparent;
        color: #4A4A4A;
        font-weight: 600;
        font-size: 1rem;
        padding: 10px 20px;
        flex-grow: 1;
        justify-content: center;
    }}
    .stTabs [aria-selected="true"] {{
        background-color: #003366 !important;
        color: #FFFFFF !important;
        border-bottom: 3px solid #FF9933;
    }}
    
    /* Section Headers */
    .section-header {{
        background: linear-gradient(135deg, #E3F2FD 0%, #DBEAFE 100%);
        border-left: 4px solid #003366;
        padding: 14px 24px;
        border-radius: 12px;
        color: #003366;
        margin-top: 28px;
        margin-bottom: 24px;
        font-weight: 600;
        font-size: 1.1rem;
        display: flex;
        align-items: center;
        box-shadow: 0 2px 12px rgba(0, 51, 102, 0.08);
    }}

    </style>
    {header_html}
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



def generate_pdf_report(kpis, recs, insights, scope_name):
    """
    Generates a professional 'Commander's Brief' PDF.
    """
    pdf = FPDF()
    pdf.add_page()
    
    # 1. Header
    pdf.set_fill_color(0, 51, 102) # Navy Blue
    pdf.rect(0, 0, 210, 40, 'F')
    
    pdf.set_font("Arial", 'B', 16)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(0, 10, "UIDAI EXECUTIVE STRATEGIES & OPERATIONAL DOSSIER", ln=True, align='C')
    pdf.set_font("Arial", '', 10)
    pdf.cell(0, 10, f"Scope: {scope_name} | Generated: {datetime.now().strftime('%d %b %Y %H:%M')}", ln=True, align='C')
    
    pdf.set_y(45)
    pdf.set_text_color(220, 38, 38) # Red
    pdf.set_font("Arial", 'B', 10)
    pdf.cell(0, 10, "CONFIDENTIAL // TYPE 2 // INTERNAL OPERATIONS ONLY", border=1, ln=True, align='C')
    
    # 2. Strategic Situation Report
    pdf.set_y(60)
    pdf.set_text_color(15, 23, 42)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "1. STRATEGIC SITUATION REPORT", ln=True)
    pdf.ln(2)
    
    pdf.set_font("Arial", '', 10)
    metrics = [
        ("Total Activity Volume", str(kpis['total'])),
        ("Daily Operational Avg", str(kpis['daily_avg'])),
        ("Backlog Clearance Est.", str(kpis.get('backlog', 'Normal'))),
        ("Child Enrolment Ratio", f"{kpis['child_pct']}%")
    ]
    
    for label, val in metrics:
        pdf.set_font("Arial", 'B', 10)
        pdf.cell(50, 8, f"{label}:", border=0)
        pdf.set_font("Arial", '', 10)
        pdf.cell(0, 8, val, ln=True)
    
    # 3. Recommendations
    pdf.ln(5)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "2. RAPID RESPONSE PROTOCOLS", ln=True)
    pdf.ln(2)
    
    for rec in recs:
        pdf.set_font("Arial", 'B', 10)
        pdf.set_fill_color(248, 250, 252)
        pdf.cell(0, 8, f"[{rec['type'].upper()}] {rec['action']}", ln=True, fill=True)
        pdf.set_font("Arial", '', 9)
        # Handle HTML tags for PDF
        detail_clean = rec['detail'].replace("<br>", "\n").replace("<b>", "").replace("</b>", "")
        pdf.multi_cell(0, 6, detail_clean)
        pdf.ln(2)
        
    # 4. Intelligence Summary
    pdf.ln(5)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "3. AUTOMATED INTELLIGENCE SUMMARY", ln=True)
    pdf.ln(2)
    
    pdf.set_font("Arial", '', 9)
    for insight in insights:
        pdf.multi_cell(0, 6, f"- {insight['title']}: {insight['text']}")
        pdf.ln(1)
        
    pdf.set_y(-20)
    pdf.set_font("Arial", 'I', 8)
    pdf.set_text_color(100, 116, 139)
    pdf.cell(0, 10, "AUTHORIZED PERSONNEL ONLY | UIDAI DATA GOVERNANCE ACT", align='C')
    
    return pdf.output()

def generate_project_docs_pdf():
    """
    Generates the 'Ultimate Project Guide' as a downloadable PDF.
    """
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "ULTIMATE PROJECT GUIDE: UIDAI ANALYTICS COMMAND CENTER", ln=True, align='C')
    pdf.set_font("Arial", 'B', 12)
    pdf.set_text_color(100, 116, 139)
    pdf.cell(0, 10, "Complete Technical & Operational Blueprint", ln=True, align='C')
    pdf.ln(10)
    
    sections = [
        ("1. PROJECT INTRODUCTION", "Systems designed to monitor national enrolment telemetry, identify bottlenecks, and optimize resource deployment across 28 States and 8 Union Territories."),
        ("2. MATHEMATICAL FORMULARY", "Includes PSACI (weighted spatial pressure), Holt-Linear Forecasting, and Z-Score Anomaly detection (3-sigma threshold)."),
        ("3. TECHNICAL ARCHITECTURE", "Python-based statistical engine utilizing NumPy and Pandas for high-speed matrix computation. Built on a Streamlit front-end and Dockerized for maximum security."),
        ("4. DATA PRIVACY", "Privacy-by-Design mandates ensuring no PII (Aadhaar, Names, Biometrics) is ever processed. Uses only aggregated transaction counts.")
    ]
    
    for title, text in sections:
        pdf.set_font("Arial", 'B', 12)
        pdf.set_text_color(0, 51, 102)
        pdf.cell(0, 10, title, ln=True)
        pdf.set_font("Arial", '', 10)
        pdf.set_text_color(15, 23, 42)
        pdf.multi_cell(0, 6, text)
        pdf.ln(4)
        
    return pdf.output()

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
            <h1>UIDAI EXECUTIVE STRATEGIES & OPERATIONAL DOSSIER</h1>
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
        bg_color = "#fffcf2"
        border_color = "#f59e0b"
        if rec['type'].upper() == "CRITICAL":
            bg_color = "#fef2f2"
            border_color = "#ef4444"
        elif rec['type'].upper() == "SURGE":
            bg_color = "#f0f9ff"
            border_color = "#0ea5e9"
            
        html += f"""
        <div class="rec-card" style="background-color: {bg_color}; border-left-color: {border_color};">
            <b style="color: {border_color}">[{rec['type'].upper()}] {rec['action']}</b>
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
def load_and_process_data(dataset_type="Enrolment"):
    DATA_CONFIG = {
        "Enrolment": {
            "local": "enrolment_data_main.csv",
            "url": "https://raw.githubusercontent.com/srinivas191206/Uidai-hackathon/main/enrolment_data_main.csv",
            "cols": {'age_0_5': 'age_0_5', 'age_5_17': 'age_5_17', 'age_18_greater': 'age_18_greater'}
        },
        "Biometric": {
            "local": "biometric_data_main.csv",
            "url": "https://raw.githubusercontent.com/srinivas191206/Uidai-hackathon/main/biometric_data_main.csv",
            "cols": {'bio_age_5_17': 'age_5_17', 'bio_age_17_': 'age_18_greater'}
        },
        "Demographic": {
            "local": "demographic_data_main.csv",
            "url": "https://raw.githubusercontent.com/srinivas191206/Uidai-hackathon/main/demographic_data_main.csv",
            "cols": {'demo_age_5_17': 'age_5_17', 'demo_age_17_': 'age_18_greater'}
        }
    }
    
    config = DATA_CONFIG.get(dataset_type, DATA_CONFIG["Enrolment"])
    LOCAL_PATH = config["local"]
    GITHUB_RAW_URL = config["url"]
    
    # Optimized Dtypes for high-speed loading
    dtypes = {
        'postal_state': str,
        'postal_district': str,
        'pincode': str
    }
    
    try:
        if os.path.exists(LOCAL_PATH):
            df = pd.read_csv(LOCAL_PATH, dtype=dtypes)
        else:
            df = pd.read_csv(GITHUB_RAW_URL, dtype=dtypes)
            
        df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y', errors='coerce')
        df = df.dropna(subset=['date'])
        
        # Standardize Columns
        for old_col, new_col in config["cols"].items():
            if old_col in df.columns:
                df[new_col] = df[old_col]
        
        # Ensure mapping columns exist
        for col in ['age_0_5', 'age_5_17', 'age_18_greater']:
            if col not in df.columns:
                df[col] = 0
                

                
        # 1. Total Enrolment per record
        df['total_activity'] = df['age_0_5'] + df['age_5_17'] + df['age_18_greater']
        
        # Normalize text columns
        df['postal_state'] = df['postal_state'].astype(str).str.title()
        
        # Global replacement for Jammu & Kashmir for consistent mapping/filtering
        df['postal_state'] = df['postal_state'].replace({
            'Jammu And Kashmir': 'Jammu & Kashmir',
            'Ladakh': 'Jammu & Kashmir',
            'Andaman And Nicobar Islands': 'Andaman & Nicobar'
        })
        
        df['postal_district'] = df['postal_district'].astype(str).str.title()
        
        return df
    except Exception as e:
        st.error(f"Critical Data Load Error ({dataset_type}): {e}")
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
            
        # --- GEOJSON MERGE: J&K + LADAKH ---
        # Combine distinct polygons into one MultiPolygon feature to unify the region visually and logically
        if states:
            features = states.get('features', [])
            jk = next((f for f in features if f['properties'].get('ST_NM') == 'Jammu & Kashmir'), None)
            ladakh = next((f for f in features if f['properties'].get('ST_NM') == 'Ladakh'), None)
            
            if jk and ladakh:
                # Helper to extract polygon rings
                def get_coords(f):
                    t = f['geometry']['type']
                    if t == 'Polygon': return [f['geometry']['coordinates']]
                    elif t == 'MultiPolygon': return f['geometry']['coordinates']
                    return []
                
                # Merge coordinates
                new_coords = get_coords(jk) + get_coords(ladakh)
                jk['geometry']['type'] = 'MultiPolygon'
                jk['geometry']['coordinates'] = new_coords
                
                # Remove Ladakh feature
                states['features'] = [f for f in features if f['properties'].get('ST_NM') != 'Ladakh']
            
        return states, districts
    except Exception as e:
        st.error(f"GeoJSON Load Error: {e}")
        return None, None



# --- 11. REAL ANALYTICS ENGINE (New Implementation) ---










# --- AUTOMATED STRATEGIC INSIGHTS ---


@st.cache_data
def generate_insights(df, dist_stats, selected_scope, scope_name, dataset_type="Enrolment"):
    insights = []
    
    # 1. High Level Trend - Demographic Pulse
    total_vol = df['total_activity'].sum()
    child_count = df['age_0_5'].sum()
    youth_count = df['age_5_17'].sum()
    adult_count = df['age_18_greater'].sum()
    
    # Hide child enrolment insights if the dataset doesn't support them
    if dataset_type == "Enrolment":
        child_pct = (child_count / total_vol) * 100 if total_vol > 0 else 0
        insights.append({
            "title": "Demographic Pulse (Infants)",
            "text": f"Across {scope_name}, child enrolment constitutes **{child_pct:.1f}%** of all activity. "
                    + ("This is below the recommended 15% threshold, indicating a need for targeted Anganwadi drives." if child_pct < 15 else "This indicates healthy saturation in early age groups.")
                    + "\n\n**What this means:** This metric monitors the intake of the youngest citizens (0-5 years) into the Aadhaar ecosystem. A low percentage suggests we are missing new births in this region.",
            "data": {
                "0-5 Years": f"{child_count:,.0f}",
                "5-17 Years": f"{youth_count:,.0f}",
                "18+ Years": f"{adult_count:,.0f}",
                "Total": f"{total_vol:,.0f}"
            }
        })
    else:
        # Contextual insight for Biometric/Demographic
        youth_pct = (youth_count / total_vol) * 100 if total_vol > 0 else 0
        adult_pct = (adult_count / total_vol) * 100 if total_vol > 0 else 0
        
        primary_group = "5-17 Years" if youth_count > adult_count else "18+ Years"
        
        insights.append({
            "title": f"Service Distribution Profile ({dataset_type})",
            "text": f"Analysis of {dataset_type} activity across {scope_name} shows that **{primary_group}** is the primary driver of volume. "
                    f"The split is **{youth_pct:.1f}%** for Children/Youth (5-17) and **{adult_pct:.1f}%** for Adults (18+). "
                    "This identify whether center capacity is being consumed by mandatory updates or new demographic captures."
                    "\n\n**What this means:** This profile helps administrators understand if centers are busy with 'new customers' (youth) or 'servicing existing ones' (adult updates).",
            "data": {
                "5-17 Years": f"{youth_count:,.0f}",
                "18+ Years": f"{adult_count:,.0f}",
                "Total Volume": f"{total_vol:,.0f}"
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
                    f"Traffic intensity suggests standard centers are overwhelmed. {rec_text} Priority Districts: **{', '.join(top_stressed_names)}**."
                    "\n\n**What this means:** 'High Demand' identifies districts where the number of daily visitors is significantly higher than the average, likely causing long wait times and center crowding.",
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
                    f"Unlike volatile drops, these areas effectively 'flatlined'. Prioritize audit for: **{', '.join(silent_dists['postal_district'].head(3).tolist())}**."
                    "\n\n**What this means:** 'Silent Under-Enrolment' detects areas where activity has mysteriously stopped or stayed very low, which might indicate broken machines or inactive centers that haven't been reported.",
            "data": {
                "Affected Districts": f"{len(silent_dists)}",
                "Avg Demand Score": f"{avg_silent_demand:.2f}x",
                "Avg Volatility": f"{avg_silent_volatility:.2f}",
                "Total Activity": f"{total_silent_activity:,.0f}"
            }
        })
    
    # 5. Demographic Transition Pressure (NEW)
    high_dtpi = dist_stats[dist_stats['dtpi'] > 0.6]
    if not high_dtpi.empty:
        avg_dtpi = high_dtpi['dtpi'].mean()
        insights.append({
            "title": "Adolescent-to-Adult Transition Surge",
            "text": f"**{len(high_dtpi)} districts** are showing critical transition pressure. High volumes of 5-17s are aging into the mandatory 18+ biometric update cycle. "
                    "Recommend proactive appointment scheduling for these age brackets."
                    "\n\n**What this means:** DTPI (Demographic Transition Pressure) predicts future crowds. It counts children who will soon turn 18 and need mandatory finger-print updates, allowing us to plan capacity ahead of time.",
            "data": {
                "High DTPI Districts": f"{len(high_dtpi)}",
                "Avg DTPI Ratio": f"{avg_dtpi:.2f}",
                "Status": "Upcoming Load Surge"
            }
        })
        
    # 6. Biometric Update Burden (NEW)
    heavy_bubr = dist_stats[dist_stats['bubr'] > 0.75]
    if not heavy_bubr.empty:
        insights.append({
            "title": "Biometric Update Concentration",
            "text": f"In **{len(heavy_bubr)} districts**, adult biometric updates/corrections exceed 75% of total center activity. "
                    "This indicates a 'Maintenance over Growth' phase. Optimize counters for update-specific flows."
                    "\n\n**What this means:** BUBR identifies if centers are spending all their time 'fixing old Aadhaar cards' instead of 'making new ones'. High values suggest we should set up dedicated fast-track lanes for simple updates.",
            "data": {
                "Affected Districts": f"{len(heavy_bubr)}",
                "Avg BUBR": f"{heavy_bubr['bubr'].mean():.1%}",
                "Priority Action": "Service Queue Segregation"
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
                    "This suggests 'Centre Deserts' in the remaining 90%. Recommendation: Dynamic resource deployment to peripheral pincodes."
                    "\n\n**What this means:** Concentration Risk flags areas where all Aadhaar services are clumped in one spot, forcing people from far-away villages to travel long distances. We need more mobile vans here.",
            "data": {
                "Affected Districts": f"{len(high_conc)}",
                "Avg Concentration": f"{avg_conc:.1%}",
                "Highest Concentration": f"{max_conc:.1%}",
                "Most Concentrated": f"{max_conc_dist}"
            }
        })

    return insights

# --- MAIN APP ---

# --- TOP CONTROL BAR (Filters) ---
st.markdown('<div class="controls-title">Dashboard Controls</div>', unsafe_allow_html=True)

# Five filters in a row
col_dataset, col_state, col_district, col_age, col_date = st.columns(5)

with col_dataset:
    dataset_type = st.selectbox(
        "Dataset Type",
        ["Enrolment", "Biometric", "Demographic"],
        help="Switch between activity datasets"
    )

df = load_and_process_data(dataset_type)
if df.empty:
    st.stop()
# Dashboard controls panel at the top with all 3 filters


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

with col_state:
    state_options = ["All India"] + sorted(df['postal_state'].unique().tolist())
    default_ix = st.session_state.get('selected_state_index', 0)
    
    if st.session_state.get('state_selector', 'All India') != "All India":
         sel_col, reset_col = st.columns([6, 1])
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
             st.button("⟳", help="Reset to National View", on_click=reset_national_view)
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
    if dataset_type == "Enrolment":
        age_group_options = {
            'age_0_5': '0-5 Years (Infants)',
            'age_5_17': '5-17 Years (Children/Youth)',
            'age_18_greater': '18+ Years (Adults)'
        }
    else:
        # Biometric and Demographic datasets don't have 0-5
        age_group_options = {
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

with col_date:
    min_date = df['date'].min().date()
    max_date = df['date'].max().date()
    
    date_range = st.date_input(
        "Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
        help="Filter data by specific time period"
    )

# PANEL CLOSING REMOVED AS IT WAS CREATING WHITESPACE
pass

# Apply Date Filter First (Global)
if len(date_range) == 2:
    start_date, end_date = date_range
    # Convert to datetime64[ns] to match df['date']
    mask = (df['date'].dt.date >= start_date) & (df['date'].dt.date <= end_date)
    df = df.loc[mask]
else:
    # Handle single date selection edge case if user picks one date
    if len(date_range) > 0:
        mask = (df['date'].dt.date == date_range[0])
        df = df.loc[mask]

# --- CRITICAL: CALCULATE METRICS *AFTER* DATE FILTER FOR DYNAMIC INSIGHTS ---
# This ensures that all downstream stats (Velocity, Demand Score) reflect the selected time window.
# The function is cached, so if the date range doesn't change, this is instant.
dist_stats_all = calculate_district_metrics(df)

# Apply Filters
if selected_state != "All India":
    filtered_df = df[df['postal_state'] == selected_state]
    if selected_district != "All":
        filtered_df = filtered_df[filtered_df['postal_district'] == selected_district]
        scope_name = f"{selected_district}, {selected_state}"
    else:
        scope_name = selected_state
    
    # Recalculate stats for the filtered view - OPTIMIZED
    # NEW: Filter the PRE-CALCULATED stats (which are now correct for the date range)
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
    if df['date'].nunique() < 7:
        return None # Suppress velocity for cold-start
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
# 1. CRITICAL: High Demand
hd_df = dist_stats_filtered[dist_stats_filtered['demand_score'] > 1.3].sort_values('demand_score', ascending=False)
if len(hd_df) > 0:
    top_5_items = [f"{row['postal_district']} ({row['demand_score']:.2f}x)" for _, row in hd_df.head(5).iterrows()]
    top_5_hd = " | ".join(top_5_items)
    recs.append({
        "type": "Critical",
        "icon": "CRITICAL",
        "action": "Initiate Rapid Deployment Protocol",
        "detail": f"High demand detected in {len(hd_df)} districts. Mobilize dynamic resource units to reduce wait times below 20 minutes.<br><b>TOP 5:</b> {top_5_hd}"
    })
    
# 2. WARNING: Silent Under-Enrolment
silent_df = dist_stats_filtered[dist_stats_filtered['is_silent_underenrolment']].sort_values('total_activity', ascending=True)
if len(silent_df) > 0:
    top_5_items = [f"{row['postal_district']} ({format_indian(row['total_activity'])})" for _, row in silent_df.head(5).iterrows()]
    top_5_silent = " | ".join(top_5_items)
    recs.append({
        "type": "Warning",
        "icon": "WARNING",
        "action": "Audit Silent Zones",
        "detail": f"{len(silent_df)} districts show suspiciously low activity. Dispatch vigilance teams to inspect center operational status.<br><b>TOP 5:</b> {top_5_silent}"
    })
    
# 3. SURGE: Demographic Transition Pressure (DTPI)
surge_df = dist_stats_filtered[dist_stats_filtered['dtpi'] > 0.6].sort_values('dtpi', ascending=False)
if len(surge_df) > 0:
    top_5_items = [f"{row['postal_district']} ({row['dtpi']:.2f}x)" for _, row in surge_df.head(5).iterrows()]
    top_5_surge = " | ".join(top_5_items)
    recs.append({
        "type": "Surge",
        "icon": "SURGE",
        "action": "Prepare Load Surge Capacity",
        "detail": f"{len(surge_df)} districts show an upcoming load surge (DTPI > 0.6). Prepare additional enrolment capacity for next 6-12 months.<br><b>TOP 5:</b> {top_5_surge}"
    })

# 4. Strategic: Child Enrolment Logic (Only for Enrolment dataset)
if dataset_type == "Enrolment" and child_pct < 20:
    recs.append({
        "type": "Strategic",
        "icon": "CHILD",
        "action": "Anganwadi Outreach Campaign",
        "detail": f"Child enrolment ({child_pct:.1f}%) is below target (25%). Partner with WCD Ministry for school/anganwadi camp drives."
    })

# 5. NEW: Biometric Update Burden (BUBR)
bubr_heavy_count = len(dist_stats_filtered[dist_stats_filtered['bubr'] > 0.75])
if bubr_heavy_count > 0:
    recs.append({
        "type": "Operational",
        "icon": "QUEUE",
        "action": "Separate Service Queues",
        "detail": f"{bubr_heavy_count} districts are 'Correction-Heavy' (BUBR > 75%). Recommend separating 'New Enrolment' vs 'Correction' queues."
    })

# --- OFFICIAL HEADER ---
# Text header removed as per user request (head.jpg is now the banner)
# st.markdown("""...""", unsafe_allow_html=True)
st.markdown(f"**Current Scope:** {scope_name} | **Data Range:** {df['date'].min().date()} to {df['date'].max().date()}")

# TABS
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Overview", 
    "District Analytics", 
    f"{dataset_type} Insights", 
    "Geographic Access", 
    "Automated Strategic Insights", 
    "Recommendations"
])

# --- TAB 1: OVERVIEW ---
with tab1:
    st.markdown('<div class="section-header">National / State Overview</div>', unsafe_allow_html=True)
    
    # Statistical Anomaly Detection
    anomalies = detect_statistical_anomalies(filtered_df)
    
    if anomalies:
        with st.expander(f"Active Incident Monitoring: {scope_name} ({len(anomalies)})", expanded=True):
            st.markdown("<div style='font-size: 0.85rem; color: #64748B; margin-bottom: 10px;'>High-confidence statistical anomalies identified via Rolling Z-Scores. These detections trigger automated operational workflows.</div>", unsafe_allow_html=True)
            
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
            
            # Limit to top 5 anomalies
            top_anomalies = anomalies[:5]
            for a in top_anomalies:
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
            
            if len(anomalies) > 5:
                st.info(f"Operational Brief: {len(anomalies) - 5} additional minor anomalies detected and logged for internal review.")

            

    else:
        st.success("System Status: Stable. No operational anomalies detected in the current window.")
    
    
    # Metrics
    # Metrics
    tot_act = total_enrolment # Alias for tab compatibility
    avg_act = filtered_df.groupby('date')['total_activity'].sum().mean()
    child_share = child_pct
    
    c1, c2, c3, c4 = st.columns(4)

    # Velocity indicators
    v_arrow = "" # Removed emoji
    v_color = "#16A34A" if global_velocity >= 0 else "#EF4444"
    velocity_html = f"<div style='color: {v_color}; font-size: 0.8rem; font-weight: 700;'>{abs(global_velocity):.1f}% Velocity</div>"

    # --- MISSION BRIEF GENERATOR ---
    # Generate data for the report
    with st.spinner("Authorizing Secure Connection..."):
        insights_preview = generate_insights(filtered_df, dist_stats_filtered, selected_state, scope_name, dataset_type)
        
        recs_preview = []
        
        # 1. CRITICAL: High Demand
        high_demand_df = dist_stats_filtered[dist_stats_filtered['demand_score'] > 1.3].sort_values('demand_score', ascending=False)
        if len(high_demand_df) > 0:
            top_5_items = [f"{row['postal_district']} ({row['demand_score']:.2f}x)" for _, row in high_demand_df.head(5).iterrows()]
            top_5_hd = " | ".join(top_5_items)
            recs_preview.append({
                "type": "Critical", 
                "icon": "CRITICAL", 
                "action": "Initiate Rapid Deployment Protocol", 
                "detail": f"High demand detected in {len(high_demand_df)} districts. Mobilize dynamic resource units to reduce wait times below 20 minutes.<br><b>TOP 5:</b> {top_5_hd}"
            })
            
        # 2. WARNING: Silent Zones
        silent_df = dist_stats_filtered[dist_stats_filtered['is_silent_underenrolment']].sort_values('total_activity', ascending=True)
        if len(silent_df) > 0:
            top_5_items = [f"{row['postal_district']} ({format_indian(row['total_activity'])})" for _, row in silent_df.head(5).iterrows()]
            top_5_silent = " | ".join(top_5_items)
            recs_preview.append({
                "type": "Warning", 
                "icon": "WARNING", 
                "action": "Audit Silent Zones", 
                "detail": f"{len(silent_df)} districts show suspiciously low activity. Dispatch vigilance teams to inspect center operational status.<br><b>TOP 5:</b> {top_5_silent}"
            })
            
        # 3. SURGE: Upcoming Transition Pressure (DTPI)
        surge_df = dist_stats_filtered[dist_stats_filtered['dtpi'] > 0.6].sort_values('dtpi', ascending=False)
        if len(surge_df) > 0:
            top_5_items = [f"{row['postal_district']} ({row['dtpi']:.2f}x)" for _, row in surge_df.head(5).iterrows()]
            top_5_surge = " | ".join(top_5_items)
            recs_preview.append({
                "type": "Surge", 
                "icon": "SURGE", 
                "action": "Prepare Load Surge Capacity", 
                "detail": f"{len(surge_df)} districts show an upcoming load surge (DTPI > 0.6). Prepare additional enrolment capacity for next 6-12 months.<br><b>TOP 5:</b> {top_5_surge}"
            })
            
        kpis_report = {
            "total": format_indian(tot_act),
            "daily_avg": format_indian(avg_act),
            "child_pct": str(round(child_share, 1))
        }
        
        
        
        # PDF Report Generator
        try:
            brief_pdf = generate_pdf_report(kpis_report, recs_preview, insights_preview, scope_name)
            st.markdown('<div style="text-align: right; margin-bottom: 10px;">', unsafe_allow_html=True)
            st.download_button(
                label="Download Executive Strategy Report (PDF Optimized)",
                data=bytes(brief_pdf),
                file_name=f"UIDAI_Executive_Report_{datetime.now().strftime('%Y%m%d')}.pdf",
                mime="application/pdf",
                key="pdf_report_btn_overview",
                help="Download a complete formal dossier including strategic insights and automated response protocols."
            )
            st.markdown('</div>', unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Report Generation Error: {e}")

    c1.markdown(f'<div class="metric-card"><div class="metric-value">{format_indian(tot_act)}</div>{velocity_html if global_velocity is not None else ""}<div class="metric-label">{dataset_type} Volume</div></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="metric-card"><div class="metric-value">{format_indian(avg_act)}</div><div class="metric-label">Daily Aggregated Avg</div></div>', unsafe_allow_html=True)
    
    if dataset_type == "Enrolment":
        c3.markdown(f'<div class="metric-card"><div class="metric-value">{child_share:.1f}%</div><div class="metric-label">Child Enrolment Ratio</div></div>', unsafe_allow_html=True)
    else:
        # Change to Youth Activity Ratio as requested
        youth_share = (filtered_df['age_5_17'].sum() / total_enrolment * 100) if total_enrolment > 0 else 0
        c3.markdown(f'<div class="metric-card"><div class="metric-value">{youth_share:.1f}%</div><div class="metric-label">Youth Activity Ratio</div></div>', unsafe_allow_html=True)
        
    c4.markdown(f'<div class="metric-card"><div class="metric-value">{len(dist_stats_filtered)}</div><div class="metric-label">Districts Active</div></div>', unsafe_allow_html=True)

    # --- STRATEGIC AWARENESS CAMPAIGN IMPACT ANALYSIS (Policy-Grade Implementation) ---
    st.markdown("---")
    st.markdown('<div class="section-header">Strategic Awareness Campaign Impact Analysis</div>', unsafe_allow_html=True)
    
    # --- 1. SETTINGS PANEL (Input Controls) ---
    
    # Calculate Data Availability (Global Scope)
    all_months_ordered = ["January", "February", "March", "April", "May", "June", 
                          "July", "August", "September", "October", "November", "December"]
                          
    if not filtered_df.empty and 'date' in filtered_df.columns:
        # Exclude 2026 data as requested
        sim_filtered_df = filtered_df[filtered_df['date'].dt.year < 2026].copy()
        
        if sim_filtered_df.empty:
            st.warning("No data available before 2026 for simulation.")
            st.stop() # Exit section if no data
            
        min_date_avail = sim_filtered_df['date'].min().date()
        max_date_avail = sim_filtered_df['date'].max().date()
        
        # Format for display (e.g., "March 2023 to Nov 2024")
        avail_text = f"{min_date_avail.strftime('%B %Y')} to {max_date_avail.strftime('%B %Y')}"
        
        # Get unique months present in data for options
        present_month_indices = sorted(sim_filtered_df['date'].dt.month.unique())
        available_month_options = [all_months_ordered[i-1] for i in present_month_indices]
        missing_months = [m for m in all_months_ordered if m not in available_month_options]
    else:
        sim_filtered_df = pd.DataFrame()
        min_date_avail = datetime.date.today()
        max_date_avail = datetime.date.today()
        avail_text = "No Data Available"
        available_month_options = []
        missing_months = all_months_ordered

    st.markdown(f"""
    <div style='background: #F8FAFC; padding: 15px; border-radius: 8px; border: 1px solid #CBD5E1; margin-bottom: 20px;'>
        <div style='display: flex; justify-content: space-between; align_items: center;'>
            <div style='font-size: 0.95em; font-weight: 600; color: #334155;'>Simulation Parameters</div>
            <div style='font-size: 0.8em; color: #64748B; background: #E2E8F0; padding: 4px 8px; border-radius: 4px;'>
                Data Available: <strong>{avail_text}</strong>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    col_sim_1, col_sim_2 = st.columns([1, 1])
    
    with col_sim_1:
        # A. Campaign Launch Month (Simulation Input)
        # Show only months with data availability as requested
        month_options_with_none = ["None (No Campaign)"] + available_month_options
        launch_month_selection = st.selectbox(
            "Campaign Launch Month",
            month_options_with_none,
            index=0,
            help="Select campaign start month or 'None'. Options restricted to months with source data."
        )
        
        # Determine launch_month_idx
        if launch_month_selection == "None (No Campaign)":
            launch_month_idx = None
            launch_month_name = "None"
            # Default to full range when no campaign
            suggested_start = min_date_avail
            suggested_end = max_date_avail
        else:
            launch_month_name = launch_month_selection
            launch_month_idx = all_months_ordered.index(launch_month_name)
            
            # Calculate FULL CALENDAR MONTH range for the selected month
            selected_month_num = launch_month_idx + 1
            
            # Find the year where this month exists in data
            month_data = sim_filtered_df[sim_filtered_df['date'].dt.month == selected_month_num]
            
            if not month_data.empty:
                # Use the year from the data
                sample_date = month_data.iloc[0]['date']
                year = sample_date.year
            else:
                # Fallback to current year if month not in data (shouldn't happen now with restriction)
                year = datetime.now().year
            
            # Calculate first and last day of the selected month
            import calendar
            first_day = datetime(year, selected_month_num, 1).date()
            last_day_num = calendar.monthrange(year, selected_month_num)[1]
            last_day = datetime(year, selected_month_num, last_day_num).date()
            
            # Ensure within available data bounds
            suggested_start = max(first_day, min_date_avail)
            suggested_end = min(last_day, max_date_avail)

    with col_sim_2:
        # B. Date Range Filter (Auto-synced to Launch Month)
        selected_date_range = st.date_input(
            "Simulation Date Range (DD/MM/YYYY)",
            value=(suggested_start, suggested_end),
            min_value=min_date_avail,
            max_value=max_date_avail,
            help="Auto-synced to selected campaign month. Shows full calendar month range.",
            format="DD/MM/YYYY"
        )
        
    # Mention missing months as a note below controls
    if missing_months:
        st.caption(f"**Note**: Source data is currently unavailable for: {', '.join(missing_months)}. The simulation excludes these periods.")
        
    # --- 2. DATA CALCULATION ---
    # Show full timeline with visual markers for selected campaign month
    
    # CRITICAL: Always use the simulation-filtered dataset (excludes 2026)
    calc_df = sim_filtered_df.copy()
    
    # Calculate impact on the full window (filtered for < 2026)
    impact_ts_full, metrics = calculate_campaign_impact_v2(calc_df, launch_month_idx)
    
    # Use full timeline for display (no filtering)
    if not impact_ts_full.empty:
        impact_ts = impact_ts_full.copy()
        
        # If user has selected a specific date range, still show full data but highlight their selection
        use_range_highlight = False
        if isinstance(selected_date_range, tuple) and len(selected_date_range) == 2:
            user_start_date, user_end_date = selected_date_range
            # Only highlight if it's not the full range
            if user_start_date != min_date_avail or user_end_date != max_date_avail:
                use_range_highlight = True
    else:
        impact_ts = pd.DataFrame()
        use_range_highlight = False
    
    # Check if final view has data
    if impact_ts.empty and not impact_ts_full.empty:
        st.warning(f"No data in the selected view range. The full simulation has {len(impact_ts_full)} data points, but your filters excluded all of them.")
    elif impact_ts.empty:
        st.info("No data available for simulation. Please check your global filters.")
    
    # --- 3. GRAPH VISUALIZATION ---
    if not impact_ts.empty:
        fig = go.Figure()
        
        # Line 1: Natural Demand
        fig.add_trace(go.Scatter(
            x=impact_ts['date'],
            y=impact_ts['natural_demand'],
            mode='lines',
            name='Natural Demand',
            line=dict(color='#94A3B8', width=2, dash='dot'),
            hovertemplate='<b>%{x|%b %Y}</b><br>Natural: %{y:,.0f}<extra></extra>'
        ))
        
        # Line 2: Projected Demand
        fig.add_trace(go.Scatter(
            x=impact_ts['date'],
            y=impact_ts['projected_demand'],
            mode='lines+markers',
            name='Projected Demand',
            line=dict(color='#003366', width=3),
            marker=dict(size=6),
            hovertemplate='<b>%{x|%b %Y}</b><br>Projected: %{y:,.0f}<br>Lift: +%{customdata:,.0f}<extra></extra>',
            customdata=impact_ts['lift']
        ))
        
        # Add visual marker for campaign launch month (peak month)
        if launch_month_idx is not None and 'natural_peak_month' in metrics:
            # Find peak date in the visible timeline
            peak_month_num = metrics['natural_peak_month'] + 1
            peak_data = impact_ts[impact_ts['date'].dt.month == peak_month_num]
            
            if not peak_data.empty:
                # Use the first occurrence of the peak month
                peak_date = peak_data.iloc[0]['date']
                peak_month_label = all_months_ordered[metrics['natural_peak_month']]
                
                # Add vertical line using add_shape (more compatible with datetime)
                fig.add_shape(
                    type="line",
                    x0=peak_date, x1=peak_date,
                    y0=0, y1=1,
                    yref="paper",
                    line=dict(color='#EF4444', width=2, dash='dash')
                )
                
                # Add annotation text separately
                fig.add_annotation(
                    x=peak_date,
                    y=1,
                    yref="paper",
                    text=f"Natural Peak ({peak_month_label})",
                    showarrow=False,
                    yshift=10,
                    font=dict(color='#EF4444', size=12)
                )
        
        # Add visual highlighting of selected date range if specified
        if use_range_highlight:
            user_start_ts = pd.Timestamp(user_start_date)
            user_end_ts = pd.Timestamp(user_end_date)
            
            # Add shaded region for selected period
            fig.add_vrect(
                x0=user_start_ts, x1=user_end_ts,
                fillcolor="#003366", opacity=0.08,
                layer="below", line_width=0,
                annotation_text="Selected Range", annotation_position="top left"
            )
        
        # Aesthetics with auto-scaled Y-axis
        fig.update_layout(
            title="Projected vs. Natural Enrolment Demand (Full Timeline)",
            xaxis_title="Timeline",
            yaxis_title="Enrolment Volume",
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=20, r=20, t=50, b=30),
            height=450,
            template="plotly_white",
            yaxis=dict(
                autorange=True,  # Auto-scale based on visible data
                rangemode='tozero'  # Start from zero for context
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Informational Disclosure Checklist
        st.markdown("""
        <div style='background: #F0F9FF; border-left: 4px solid #0EA5E9; padding: 12px 18px; border-radius: 8px; margin-bottom: 25px;'>
            <div style='display: flex; align-items: start; gap: 12px;'>
                <div style='color: #0284C7; font-size: 1.4rem; font-weight: 700;'>ℹ️</div>
                <div>
                    <div style='color: #0369A1; font-weight: 700; font-size: 1rem; margin-bottom: 4px;'>What does this mean?</div>
                    <div style='color: #475569; font-size: 0.9rem; line-height: 1.5;'>
                        This simulator estimates relative uplift under different campaign lead-times. 
                        It does not predict exact enrolment counts but helps compare timing strategies.
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # --- 4. STATE INSIGHT PANEL ---
        st.markdown("### Strategic Insight Panel")
        
        col_ins_1, col_ins_2 = st.columns([1, 2])
        
        with col_ins_1:
             # Using separate metrics for cleaner layout (Fixing HTML issue)
             st.markdown(f"**Analysis Scope:** {selected_state}")
             
             st.markdown(f"""
             <div style='background: #F1F5F9; padding: 12px; border-radius: 8px; margin-top: 5px;'>
                <div style='margin-bottom: 8px;'>
                    <div style='font-size: 0.75rem; color: #64748B;'>Natural Peak</div>
                    <div style='font-weight: 600; color: #334155;'>{["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"][metrics['natural_peak_month']]}</div>
                </div>
                <div style='margin-bottom: 8px;'>
                    <div style='font-size: 0.75rem; color: #64748B;'>Lead Time</div>
                    <div style='font-weight: 600; color: #334155;'>{metrics['lead_time']} months</div>
                </div>
                <div style='margin-bottom: 8px;'>
                    <div style='font-size: 0.75rem; color: #64748B;'>Amp Factor (Ab)</div>
                    <div style='font-weight: 600; color: #334155;'>{metrics['ab']:.2f}</div>
                </div>
                <div>
                    <div style='font-size: 0.75rem; color: #64748B;'>Selected Launch</div>
                    <div style='font-weight: 600; color: #0284C7;'>{launch_month_name}</div>
                </div>
             </div>
             """, unsafe_allow_html=True)
            
        with col_ins_2:
            # Metrics Logic
            paf = metrics['paf']
            ori = metrics['ori']
            
            # Interpretation
            if ori > 1.0:
                status = "High Operational Risk"
                color = "#DC2626" # Red
                desc = "Projected demand likely to exceed infrastructure capacity. Breakdown imminent."
            elif ori >= 0.9:
                status = "Warning - Near Capacity"
                color = "#F59E0B" # Orange
                desc = "Operating within safety margins but with elevate strain. Monitor carefully."
            else:
                status = "Safe - Capacity Adequate"
                color = "#16A34A" # Green
                desc = "Projected demand is well within operational limits. Launch recommended."
                
            st.markdown(f"""
            <div style='display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-bottom: 15px;'>
                <div style='border: 1px solid #E2E8F0; padding: 10px; border-radius: 6px; text-align: center;'>
                    <div style='font-size: 0.75rem; color: #64748B; text-transform: uppercase;'>Peak Amplification (PAF)</div>
                    <div style='font-size: 1.5rem; font-weight: 700; color: #334155;'>{paf:+.1%}</div>
                </div>
                <div style='border: 1px solid {color}40; background: {color}05; padding: 10px; border-radius: 6px; text-align: center;'>
                    <div style='font-size: 0.75rem; color: {color}; text-transform: uppercase;'>Ops Risk Index (ORI)</div>
                    <div style='font-size: 1.5rem; font-weight: 700; color: {color};'>{ori:.2f}</div>
                </div>
            </div>
            <div style='background: {color}10; border-left: 4px solid {color}; padding: 12px 16px; border-radius: 4px;'>
                <div style='font-weight: 700; color: {color}; margin-bottom: 4px;'>{status}</div>
                <div style='font-size: 0.9rem; color: #334155;'>{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    else:
        st.info("No data available for the selected range/filters to generate impact analysis.")
    
    st.markdown("---")

    # --- SIMULATED SYSTEM BOOT ---
    # --- SYSTEM STATUS INDICATOR ---
    if 'system_ready' not in st.session_state:
        st.session_state['system_ready'] = True
    
    # --- POLICY SIMULATOR (DECISION SUPPORT) ---
    with st.expander("Decision Support System (Policy Simulator)", expanded=True):
        st.markdown("##### Dynamic Enrolment Capacity Rebalancing System")
        sim_col1, sim_col2, sim_col3 = st.columns(3)
        
        with sim_col1:
            capacity_adj_count = st.number_input("Capacity Adjustment (People)", min_value=0, max_value=50, value=0, step=1)
            st.caption("Additional personnel to deploy. Max 5 people per center across ~10 centers.")
            
        with sim_col2:
            extra_hours = st.number_input("Extend Center Hours", min_value=0, max_value=4, value=0, step=1)
            st.caption("Additional operational hours per day.")
            
        # Calculate Impact
        current_daily_capacity = avg_act # Using active daily avg as proxy for capacity
        current_backlog = tot_act * 0.15 # Simulation: 15% hidden backlog
        
        # Use the people count directly as additional capacity units
        # Each person/counter can handle ~80 enrolments per day
        added_capacity_units = capacity_adj_count
        
        impact = simulate_policy_impact(current_daily_capacity, current_backlog, added_capacity_units, extra_hours)
        
        with sim_col3:
            if capacity_adj_count > 0 or extra_hours > 0:
                st.markdown(f"""
                    <div style='background-color: #F0FDF4; border: 1px solid #16A34A; border-radius: 8px; padding: 10px; text-align: center;'>
                        <div style='color: #166534; font-size: 0.8rem; font-weight: 600;'>PROJECTED CLEARANCE</div>
                        <div style='color: #15803D; font-size: 1.4rem; font-weight: 800;'>{impact['days_to_clear_new']:.1f} Days</div>
                        <div style='color: #166534; font-size: 0.8rem;'>Saved: {impact['days_saved']:.1f} Days</div>
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
        '> Strategic Analytics & Forecasting</span>
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
                          title="Temporal Trends and Predictive Analytics", color_discrete_sequence=['#2563EB', '#F59E0B', '#10B981'])
        
        # Add REAL Prediction (Holt-Winters Approx)
        if len(daily) > 5:
            forecast_vals = calculate_forecast_holt_winters(daily['total_activity'].tolist(), n_preds=15)
            last_date = daily['date'].max()
            future_dates = pd.date_range(start=last_date, periods=16)[1:]
            
            fig_line.add_scatter(x=future_dates, y=forecast_vals, mode='lines', 
                                name='Predictive Analytics (Trend+Seasonality)', 
                                line=dict(color='#FF0000', width=2, dash='dot'))
            
            # Policy Simulation Impact overlay
            if capacity_adj_count > 0 or extra_hours > 0:
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
                    'The Dadra And Nagar Haveli And Daman And Diu': 'Dadra and Nagar Haveli and Daman and Diu'
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
                    fig_map.update_geos(
                        visible=False, 
                        projection_scale=1.0,
                        lataxis_range=[6.5, 37.5], 
                        lonaxis_range=[67.0, 98.0]
                    )
                    # Enable Selection
                    fig_map.update_layout(clickmode='event+select', height=850, margin={"r":0,"t":0,"l":0,"b":20},
                                         coloraxis_colorbar=dict(title="Enrolments", len=0.4), autosize=True)
                    
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
                        'Jammu & Kashmir': (33.7782, 76.5762),
                        'Himachal Pradesh': (31.1048, 77.1734),
                        'Uttarakhand': (30.0668, 79.0193),
                        'Arunachal Pradesh': (28.2180, 94.7278),
                        'Manipur': (24.6637, 93.9063),
                        'Meghalaya': (25.4670, 91.3662),
                        'Mizoram': (23.1645, 92.9376),
                        'Nagaland': (26.1584, 94.5624),
                        'Tripura': (23.9408, 91.9882),
                        'Sikkim': (27.5330, 88.5122),
                        'Andaman & Nicobar': (11.7401, 92.6586),
                        'Dadra and Nagar Haveli and Daman and Diu': (20.1809, 73.0169),
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
                        textfont=dict(size=10, color='#000000', family='Poppins, sans-serif'), # Fix applied
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
                        featureidkey='properties.NAME_2',
                        locations='postal_district',
                        color='total_activity',
                        color_continuous_scale=custom_color_scale,
                        title=f"District Activity in {selected_state}",
                        hover_name='postal_district',
                        labels={'total_activity': 'Enrolments'}
                    )
                     fig_map.update_geos(
                         visible=False, 
                         projection_scale=1.0, 
                         fitbounds="locations"
                     )
                     fig_map.update_layout(height=850, margin={"r":0,"t":0,"l":0,"b":20},
                                         coloraxis_colorbar=dict(title="Enrolments", len=0.4), autosize=True)
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
            hover_data={'stress_persistence_months': True, 'volatility_score': True},
            labels={'stress_persistence_months': 'Peak Load Persistence', 'volatility_score': 'Volatility Score'},
            title="District Performance Clusters (Algorithmic Segmentation)",
            color_discrete_map={
                'Critical Care Zone': '#EF4444', 
                'Monitoring Zone': '#F59E0B', 
                'Stable Zone': '#10B981'
            }
        )
        st.plotly_chart(fig_map, use_container_width=True)
        
    with col_d2:
        st.subheader("High Demand & Pressure Districts")
        st.markdown("Districts exceeding 1.3× National Average or showing Demographic Surge")
        
        # Merge classifications for table display
        table_df = dist_stats_filtered.sort_values(['demand_score', 'dtpi'], ascending=False).head(15).copy()
            
        st.dataframe(
            table_df[['postal_district', 'demand_score', 'dtpi', 'bubr']],
            column_config={
                "postal_district": st.column_config.TextColumn(
                    "District",
                    help="Official administrative district name"
                ),
                "demand_score": st.column_config.NumberColumn(
                    "Demand X",
                    help="Ratio of district activity vs national average (1.0 = baseline average)",
                    format="%.2f"
                ),
                "dtpi": st.column_config.NumberColumn(
                    "Transition Index",
                    help="Demographic Transition Pressure Index (Youth vs Adult ratio). >0.6 indicates upcoming load surge.",
                    format="%.2f"
                ),
                "bubr": st.column_config.NumberColumn(
                    "Update Burden",
                    help="Share of adult biometric corrections/updates relative to total district activity.",
                    format="%.1%"
                )
            },
            hide_index=True
        )
        
        # Explanatory info
        st.markdown(""" 
        <div style='background: #F0F9FF; border-left: 3px solid #0EA5E9; padding: 12px 16px; border-radius: 8px; margin-top: 16px; font-size: 0.85rem;'>
            <strong style='color: #0369A1;'>Understanding Advanced Indices</strong><br>
            <span style='color: #475569;'><b>Demand Score:</b> Ratio of district activity vs national average (1.0 = average)<br>
            <b>DTPI:</b> Transition Pressure (Youth vs Adult ratio). >0.6 indicates upcoming load surge.<br>
            <b>BUBR:</b> Update Burden. Share of adult biometric corrections/updates.</span>
        </div>
        """, unsafe_allow_html=True)

# --- TAB 3: AGE DEEP DIVE ---
with tab3:
    st.markdown(f'<div class="section-header">{dataset_type} Segment Analysis</div>', unsafe_allow_html=True)
    
    c_age1, c_age2 = st.columns(2)
    
    with c_age1:
        if dataset_type == "Enrolment":
            st.subheader("Child Enrolment Analysis")
            st.markdown("Districts with lowest proportionate child enrolment")
            laggards = dist_stats_filtered.sort_values('child_ratio', ascending=True).head(15)
            fig_lag = px.bar(laggards, x='child_ratio', y='postal_district', orientation='h', 
                            title="Districts with Lowest Child Enrolment Ratio", color='child_ratio', color_continuous_scale='Reds_r')
            st.plotly_chart(fig_lag, use_container_width=True)
            
            # Explanatory info
            st.markdown(""" 
            <div style='background: #FEF3C7; border-left: 3px solid #F59E0B; padding: 12px 16px; border-radius: 8px; font-size: 0.85rem;'>
                <strong style='color: #B45309;'>What is Child Enrolment Ratio?</strong><br>
                <span style='color: #475569;'>The percentage of 0-5 year age group enrolments compared to total district activity. 
                Low values may indicate lower participation from young children.</span>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.subheader("Youth Activity Analysis")
            st.markdown(f"Districts with lowest proportionate youth (5-17) {dataset_type} activity")
            laggards = dist_stats_filtered.sort_values('youth_ratio', ascending=True).head(15)
            fig_lag = px.bar(laggards, x='youth_ratio', y='postal_district', orientation='h', 
                            title=f"Districts with Lowest Youth Ratio ({dataset_type})", color='youth_ratio', color_continuous_scale='Purples_r')
            st.plotly_chart(fig_lag, use_container_width=True)
            
            # Explanatory info
            st.markdown(f""" 
            <div style='background: #F5F3FF; border-left: 3px solid #8B5CF6; padding: 12px 16px; border-radius: 8px; font-size: 0.85rem;'>
                <strong style='color: #6D28D9;'>What is Youth Activity Ratio?</strong><br>
                <span style='color: #475569;'>The percentage of 5-17 year age group {dataset_type} activity compared to total activity in a district. 
                This identifies regions where mandatory biometric updates for youth may be lower than expected.</span>
            </div>
            """, unsafe_allow_html=True)

    # --- TRIVARIATE ANALYSIS ---
    st.markdown("---")
    st.subheader("Trivariate Analysis")
    
    heatmap_desc = "Demographic Heatmap Matrix: Visualizing activity intensity across districts and time, colored by Child Enrolment Ratio." if dataset_type == "Enrolment" else f"Demographic Heatmap Matrix: Visualizing activity intensity across districts and time, colored by Youth {dataset_type} Ratio."
    st.markdown(heatmap_desc)
    
    # Prepare Data for Heatmap
    # Metric: Child Ratio over Time per District (Top 20 Districts by volume)
    top_districts = dist_stats_filtered.nlargest(20, 'total_activity')['postal_district'].tolist()
    tri_df = filtered_df[filtered_df['postal_district'].isin(top_districts)].copy()
    tri_df['month'] = tri_df['date'].dt.strftime('%Y-%m')
    
    heatmap_data = tri_df.groupby(['postal_district', 'month']).agg(
        total=('total_activity', 'sum'),
        child=('age_0_5', 'sum'),
        youth=('age_5_17', 'sum')
    ).reset_index()
    
    if dataset_type == "Enrolment":
        heatmap_data['ratio'] = (heatmap_data['child'] / (heatmap_data['total'] + 1e-9)) * 100
        z_label = "Child Ratio %"
        title_text = "Child Enrolment Intensity Matrix (Red=Low, Blue=High)"
    else:
        heatmap_data['ratio'] = (heatmap_data['youth'] / (heatmap_data['total'] + 1e-9)) * 100
        z_label = f"Youth {dataset_type} Ratio %"
        title_text = f"Youth {dataset_type} Intensity Matrix (Red=Low, Blue=High)"
    
    fig_heat = px.density_heatmap(
        heatmap_data, 
        x='month', 
        y='postal_district', 
        z='ratio', 
        histfunc='avg',
        title=title_text,
        color_continuous_scale='RdBu',
        labels={'postal_district': 'District', 'month': 'Month', 'ratio': z_label}
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
            <strong style='color: #1D4ED8;'>What is Adult Update Load Index?</strong><br>
            <span style='color: #475569;'>Measures the proportion of adult activities (updates, corrections, renewals) 
            relative to total district operations. High values indicate centres are primarily handling adult updates, 
            which may require additional staffing resources.</span>
        </div>
        """, unsafe_allow_html=True)

# --- TAB 4: PINCODE HEATMAP ---
with tab4:
    st.markdown('<div class="section-header">Geographic Access Distribution</div>', unsafe_allow_html=True)
    
    st.markdown(""" 
    <div style='background: #F5F3FF; border-left: 3px solid #8B5CF6; padding: 14px 18px; border-radius: 8px; margin-bottom: 20px;'>
        <strong style='color: #6D28D9; font-size: 1rem;'>Pincode Service Access Concentration Index (PSACI)</strong><br>
        <span style='color: #475569; font-size: 0.9rem;'>PSACI is a composite index (0 to 1) measuring <b>"Service Friction"</b>. It weight-aggregates demand intensity (40%), child enrolment imbalance (30%), and operational volatility (30%). 
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
            <strong>Composite Formula:</strong><br>
            PSACI = (Demand Score × 0.4) + (Child Imbalance × 0.3) + (Volatility × 0.3)<br><br>
            <strong>Interpretation Thresholds:</strong><br>
            <b>> 0.8:</b> Systemic Blindspot – Infrastructure deficit.<br>
            <b>0.5 - 0.8:</b> Service Friction – Needs monitoring.<br>
            <b>< 0.5:</b> Equitable Access – Normal operations.
        </div>
        """, unsafe_allow_html=True)
    
    
    # Additional explanation
    st.markdown(""" 
    <div style='background: #ECFDF5; border-left: 3px solid #10B981; padding: 12px 16px; border-radius: 8px; margin-top: 16px; font-size: 0.85rem;'>
        <strong style='color: #047857;'>Recommended Action</strong><br>
        <span style='color: #475569;'>Districts with concentration risk >60% should be prioritized for dynamic resource deployment 
        to peripheral pincodes. This helps ensure equitable access to Aadhaar services across all geographic areas.</span>
    </div>
    """, unsafe_allow_html=True)

# --- TAB 5: AUTOMATED INSIGHTS ---
with tab5:
    st.markdown('<div class="section-header">Automated Policy Narrative Engine</div>', unsafe_allow_html=True)
    
    insights = generate_insights(filtered_df, dist_stats_filtered, selected_state, scope_name, dataset_type)
    
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
    
    
    

    
    metric_cols = st.columns(4)
    with col_kpi:
        st.markdown("### Methodology Reference")
        
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
            <b>Formula:</b> Count of months where demand exceeds 1.3x national average<br>
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
        



