import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests
import json
import base64
from datetime import datetime as dt

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
        background-color: white;
        border-bottom: 2px solid #003366;
    }}
    .fixed-header img {{
        width: 100%;
        height: auto;
        display: block;
        max-height: 120px;
        object-fit: cover;
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
        margin-top: 120px;
        height: calc(100vh - 120px);
        background-color: #FDFDFD;
        border-right: 1px solid #E2E8F0;
    }}

    .block-container {{
        padding-top: 140px !important;
    }}
    </style>
    
    <div class="fixed-header">
        <img src="data:image/jpg;base64,{img}">
        <div style="position: absolute; bottom: 5px; right: 20px; color: #003366; font-size: 0.7rem; font-family: monospace; letter-spacing: 1.5px; opacity: 0.6;">
            SECURE CIDR GATEWAY // OPS-INTEL-2.4 // SESSION: {dt.now().strftime('%Y%j%H')}
        </div>
    </div>
""", unsafe_allow_html=True)
