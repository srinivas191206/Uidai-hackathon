import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

# --- 11. REAL ANALYTICS ENGINE (New Implementation) ---

@st.cache_data
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

@st.cache_data
def detect_statistical_anomalies(df, window=7):
    """
    Identifies high-confidence operational anomalies using rolling Z-scores.
    Logic: Z = (x - mean) / std. Flags deviations > 3 sigma.
    Interpretation: Spike/Drop indicating infrastructure or system issues.
    """
    daily = df.groupby('date')['total_activity'].sum().reset_index()
    daily = daily.sort_values('date')
    
    # --- COLD-START CONFIDENCE SUPPRESSION ---
    if len(daily) < 7:
        return [] # Suppress Z-score alerts for cold-start periods
        
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

@st.cache_data
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


def calculate_temporal_impact(df, policy_date):
    """
    Computes policy impact over a 14-day pre/post window.
    Impact = (PostAvg - PreAvg) / PreAvg * 100
    """
    policy_dt = pd.to_datetime(policy_date)
    pre_window = df[(df['date'] >= policy_dt - pd.Timedelta(days=14)) & (df['date'] < policy_dt)]
    post_window = df[(df['date'] >= policy_dt) & (df['date'] < policy_dt + pd.Timedelta(days=14))]
    
    pre_avg = pre_window.groupby('date')['total_activity'].sum().mean() if not pre_window.empty else 0
    post_avg = post_window.groupby('date')['total_activity'].sum().mean() if not post_window.empty else 0
    
    impact = ((post_avg - pre_avg) / (pre_avg + 1e-9)) * 100 if pre_avg > 0 else 0
    
    return {
        "pre_avg": pre_avg,
        "post_avg": post_avg,
        "impact_pct": impact,
        "pre_data": pre_window.groupby('date')['total_activity'].sum().reset_index(),
        "post_data": post_window.groupby('date')['total_activity'].sum().reset_index()
    }


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
        if score > 1.3: return 'High (Red)'
        elif score < 0.8: return 'Low (Blue)'
        else: return 'Medium (Yellow)'
    dist_stats['demand_zone'] = dist_stats['demand_score'].apply(classify_zone)
    
    # 4. Child Enrolment Ratio
    dist_stats['child_ratio'] = dist_stats['age_0_5'] / dist_stats['total_activity']
    
    # 4b. Youth Enrolment Ratio (Proxy for younger demographics in Biometric/Demographic datasets)
    dist_stats['youth_ratio'] = dist_stats['age_5_17'] / dist_stats['total_activity']
    
    # 5. Update Pressure Index (Approximated as 18+ activity share)
    dist_stats['update_pressure'] = dist_stats['age_18_greater'] / dist_stats['total_activity']
    
    # 6. Age Mix Imbalance Index
    dist_stats['age_mix_imbalance'] = dist_stats['age_18_greater'] / (dist_stats['age_0_5'] + dist_stats['age_5_17'])
    
    # 7. Volatility Score & 9. Stress Persistence
    # Optimization: Use 'month' directly if possible, avoiding repeated string conversion
    month_series = df['date'].dt.to_period('M')
    monthly = df.assign(month=month_series).groupby(['postal_district', 'month'])['total_activity'].sum().reset_index()
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
    
    # 8. Pincode Concentration Risk (VECTORIZED)
    # Calculate for all pincodes at once
    pin_totals = df.groupby(['postal_district', 'pincode'])['total_activity'].sum().reset_index()
    pin_totals = pin_totals.sort_values(['postal_district', 'total_activity'], ascending=[True, False])
    
    # Group by district and calculate top 10% share
    def get_top_10_share(group):
        total = group['total_activity'].sum()
        if total == 0: return 0
        n_top = max(1, int(len(group) * 0.1))
        return group['total_activity'].head(n_top).sum() / (total + 1)

    conc_df = pin_totals.groupby('postal_district').apply(get_top_10_share).reset_index(name='concentration_risk')
    dist_stats = dist_stats.merge(conc_df, on='postal_district', how='left')

    # --- NEW ANALYTICAL MODULES (DTPI, BUBR, EQUITY RISK) ---
    # Safe division for DTPI: handle cases where adult activity is zero to prevent infinite values
    dist_stats['dtpi'] = np.where(
        dist_stats['age_18_greater'] > 0,
        dist_stats['age_5_17'] / dist_stats['age_18_greater'],
        dist_stats['age_5_17'] # Treat as raw youth volume if no adults, or cap it (here we keep youth count as proxy)
    )
    # Normalize DTPI to a reasonable range if it explodes
    dist_stats['dtpi'] = dist_stats['dtpi'].clip(0, 10.0)
    
    dist_stats['bubr'] = dist_stats['age_18_greater'] / (dist_stats['total_activity'] + 1e-9)
    
    def classify_dtpi(val):
        if val > 0.6: return "Upcoming Load Surge"
        elif val >= 0.3: return "Stable Transition"
        else: return "Aging-Dominant Region"
        
    dist_stats['dtpi_class'] = dist_stats['dtpi'].apply(classify_dtpi)
    dist_stats['is_correction_heavy'] = dist_stats['bubr'] > 0.75
    
    # PSACI placeholder if not in dist_stats, otherwise use existing concentration_risk as proxy
    dist_stats['equity_risk_flag'] = (dist_stats['concentration_risk'] > 0.7) & (dist_stats['dtpi'] > 0.5)
    
    return dist_stats

def perform_custom_clustering(df, n_clusters=3):
    """
    Performs K-Means clustering using Numpy (No Scikit-Learn dependency).
    Used to segment districts into 'Critical', 'Stable', 'High Performance'.
    """
    if len(df) < n_clusters:
        # Fallback for single district or small count selection
        labels = []
        for _, row in df.iterrows():
            if row['demand_score'] > 1.3: labels.append('High-Intensity Operations')
            elif row['demand_score'] < 0.8: labels.append('Low-Enrolment / Outreach Needed')
            else: labels.append('Steady State / Monitoring')
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
    cluster_map[mean_scores[0][0]] = 'High-Intensity Operations' # Critical
    cluster_map[mean_scores[1][0]] = 'Steady State / Monitoring'   # Stable
    cluster_map[mean_scores[2][0]] = 'Low-Enrolment / Outreach Needed' # Low Activity
    
    return [cluster_map[label] for label in cluster_labels]

def generate_awareness_impact_data(advertising_month_idx=None):
    """
    Generates synthetic monthly demand data to demonstrate advertising impact.
    
    Args:
        advertising_month_idx (int): 0-11 representing Jan-Dec. If None, no advertising.
        
    Returns:
        pd.DataFrame: Monthly data with columns [month, natural_demand, observed_demand, impact_flag]
        dict: Impact metrics {amplification_factor, overload_risk}
    """
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    
    # 1. Base Natural Seasonal Curve (Arbitrary units, e.g., '000s)
    # Peak in May (Month 4) and Oct (Month 9) - typical academic/exam cycles
    base_demand = np.array([120, 130, 150, 180, 250, 200, 160, 150, 180, 240, 190, 140])
    
    # Add some random noise for "Government Reality" look
    np.random.seed(42) # Deterministic for consistent UI
    noise = np.random.normal(0, 5, 12)
    natural_demand = base_demand + noise
    natural_demand = np.maximum(natural_demand, 50) # Floor at 50
    
    # 2. Apply Advertising Impact
    observed_demand = natural_demand.copy()
    impact_metrics = {"amplification_factor": 0.0, "overload_risk": 0.0, "insight_type": "Neutral"}
    
    if advertising_month_idx is not None:
        adv_month = advertising_month_idx
        
        # Identify nearest future peak from ad month
        peaks = [4, 9] # May, Oct indices
        
        # Simple logic: Find usage relative to specific peaks
        # We simulate the logic requested:
        # - 2 months before peak: Smooths demand
        # - 1 month before/during: Amplifies peak
        
        # Let's verify relation to peaks
        dist_to_peak = min([(p - adv_month) for p in peaks if (p - adv_month) >= 0] or [99])
        
        # Case A: 2 Months before a peak (Best Practice)
        if dist_to_peak == 2:
            # Shift 20% of peak volume to earlier months (Ad month + next month)
            peak_idx = adv_month + 2
            shift_vol = observed_demand[peak_idx] * 0.20
            
            observed_demand[peak_idx] -= shift_vol
            observed_demand[adv_month] += shift_vol * 0.4
            observed_demand[adv_month+1] += shift_vol * 0.6
            
            impact_metrics["amplification_factor"] = -0.15 # Negative = Smoothing
            impact_metrics["insight_type"] = "Smoothing"
            
        # Case B: 1 Month before or During Peak (Bad Timing)
        elif dist_to_peak <= 1 and dist_to_peak >= 0:
            target_idx = adv_month + dist_to_peak # The peak itself
            
            # Massive spike at the peak
            spike = observed_demand[target_idx] * 0.35
            observed_demand[target_idx] += spike
            
            # Borrow from future (burnout)
            if target_idx + 1 < 12:
                observed_demand[target_idx+1] -= spike * 0.5
                
            impact_metrics["amplification_factor"] = 0.35
            impact_metrics["insight_type"] = "Amplification"
            
        else:
            # General "Base Awareness" bump if not near peak
            observed_demand[adv_month] += 20
            observed_demand[(adv_month+1)%12] += 10
            impact_metrics["insight_type"] = "General Awareness"

    # 3. Calculate Risk Score (Pk / Capacity)
    # Assume capacity is fixed at 260
    capacity = 260
    max_observed = observed_demand.max()
    impact_metrics["overload_risk"] = max_observed / capacity
    
    # Create DF
    df = pd.DataFrame({
        "Month": months,
        "Natural Demand": natural_demand,
        "Observed Demand": observed_demand
    })
    
    return df, impact_metrics
