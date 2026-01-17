import pandas as pd
import time
import sys
import os

# Add background to path
sys.path.append(os.path.join(os.getcwd(), "backend"))
from statistical_engine import calculate_district_metrics

def profile_load_and_calc(file_path):
    print(f"Profiling {file_path}...")
    start_time = time.time()
    
    # Simulate current loading in app.py
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y', errors='coerce')
    df = df.dropna(subset=['date'])
    
    # Standardize cols for calc_district_metrics (using biometric config as example)
    if 'bio_age_5_17' in df.columns:
        df['age_5_17'] = df['bio_age_5_17']
        df['age_18_greater'] = df['bio_age_17_']
    elif 'demo_age_5_17' in df.columns:
        df['age_5_17'] = df['demo_age_5_17']
        df['age_18_greater'] = df['demo_age_17_']
        
    for col in ['age_0_5', 'age_5_17', 'age_18_greater']:
        if col not in df.columns:
            df[col] = 0
    df['total_activity'] = df['age_0_5'] + df['age_5_17'] + df['age_18_greater']
        
    load_time = time.time() - start_time
    print(f"  Load and preprocessing time: {load_time:.2f}s")
    
    start_time = time.time()
    stats = calculate_district_metrics(df)
    calc_time = time.time() - start_time
    print(f"  calculate_district_metrics time: {calc_time:.2f}s")
    print(f"  Total time: {load_time + calc_time:.2f}s")
    print("-" * 30)

if __name__ == "__main__":
    files = [
        "enrolment_data_main.csv",
        "biometric_data_main.csv",
        "demographic_data_main.csv"
    ]
    for f in files:
        if os.path.exists(f):
            profile_load_and_calc(f)
        else:
            print(f"File {f} not found!")
