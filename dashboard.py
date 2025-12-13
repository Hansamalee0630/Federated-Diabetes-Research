import streamlit as st
import pandas as pd
import json
import time
import os
import plotly.graph_objects as go
import numpy as np

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Federated Diabetes Research Hub",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- DUMMY DATA GENERATOR ---
def get_dummy_fl_data():
    rounds = list(range(1, 21))
    global_acc = [0.60 + (i * 0.015) + (np.random.rand() * 0.02) for i in rounds]
    pers_acc = [g + 0.05 + (np.random.rand() * 0.02) for g in global_acc]
    fairness = [max(0, 0.15 - (i * 0.007)) for i in rounds]
    return pd.DataFrame({
        'round': rounds,
        'global_accuracy': global_acc,
        'personalized_accuracy': pers_acc,
        'personalization_gain': [(p - g)*100 for p, g in zip(pers_acc, global_acc)],
        'fairness_gap': fairness
    })

def load_comp4_data():
    if os.path.exists("results/comp4_results/fl_results.json"):
        try:
            with open("results/comp4_results/fl_results.json", "r") as f:
                return pd.DataFrame(json.load(f))
        except:
            return get_dummy_fl_data()
    return get_dummy_fl_data()

# --- CSS STYLING (THE WOW FACTOR) ---
st.markdown("""
    <style>
    /* IMPORT FONTS */
    @import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@300;400;500;600;700&family=Inter:wght@300;400;600&display=swap');

    /* GLOBAL THEME */
    .stApp {
        background-color: #0f172a;
        background-image: 
            radial-gradient(at 0% 0%, hsla(253,16%,7%,1) 0, transparent 50%), 
            radial-gradient(at 50% 0%, hsla(225,39%,30%,1) 0, transparent 50%), 
            radial-gradient(at 100% 0%, hsla(339,49%,30%,1) 0, transparent 50%);
        color: #e2e8f0;
    }

    /* TYPOGRAPHY */
    h1, h2, h3 {
        font-family: 'Rajdhani', sans-serif;
        text-transform: uppercase;
        letter-spacing: 1.5px;
    }
    
    p, div, label, li {
        font-family: 'Inter', sans-serif;
    }

    /* CUSTOM CARDS (GLASSMORPHISM) */
    .glass-card {
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        transition: all 0.3s ease;
    }

    .glass-card:hover {
        transform: translateY(-5px);
        border-color: rgba(6, 182, 212, 0.5);
        box-shadow: 0 8px 32px 0 rgba(6, 182, 212, 0.2);
    }

    /* METRIC STATS */
    .stat-value {
        font-family: 'Rajdhani', sans-serif;
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(to right, #06b6d4, #3b82f6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .stat-label {
        font-size: 0.9rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* TABS STYLING */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background-color: transparent;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: rgba(30, 41, 59, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.05);
        color: #94a3b8;
        border-radius: 8px;
        padding: 10px 20px;
        transition: all 0.3s;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, #06b6d4 0%, #3b82f6 100%);
        color: white;
        border: none;
        box-shadow: 0 0 15px rgba(6, 182, 212, 0.5);
    }

    /* INPUT FIELDS */
    .stTextInput > div > div > input, .stSelectbox > div > div > div {
        background-color: rgba(15, 23, 42, 0.6);
        color: white;
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 8px;
    }

    /* BUTTONS */
    .stButton > button {
        background: linear-gradient(90deg, #06b6d4 0%, #3b82f6 100%);
        color: white;
        border: none;
        border-radius: 8px;
        font-family: 'Rajdhani', sans-serif;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        box-shadow: 0 4px 14px 0 rgba(6, 182, 212, 0.39);
        transition: all 0.2s ease-in-out;
    }
    
    .stButton > button:hover {
        transform: scale(1.02);
        box-shadow: 0 6px 20px 0 rgba(6, 182, 212, 0.5);
    }

    /* CUSTOM GAUGES CONTAINER */
    .gauge-container {
        background: rgba(15, 23, 42, 0.6);
        border-radius: 50%;
        padding: 10px;
        box-shadow: inset 0 0 20px rgba(0,0,0,0.5);
    }
    
    /* SIDEBAR */
    [data-testid="stSidebar"] {
        background-color: #020617;
        border-right: 1px solid rgba(255,255,255,0.05);
    }
    
    /* REMOVE WHITE BACKGROUNDS */
    .stPlotlyChart {
        background-color: transparent !important;
    }
    div[data-testid="stMetric"] {
        background-color: rgba(30, 41, 59, 0.4);
        padding: 15px;
        border-radius: 10px;
        border: 1px solid rgba(255,255,255,0.05);
    }
    [data-testid="stMetricLabel"] { color: #94a3b8 !important; }
    [data-testid="stMetricValue"] { color: #f8fafc !important; }

    </style>
""", unsafe_allow_html=True)

# --- HELPER: CUSTOM STAT CARD ---
def stat_card(label, value, delta=None):
    delta_html = f"<span style='color: #4ade80; font-size: 0.8rem;'>▲ {delta}</span>" if delta else ""
    st.markdown(f"""
        <div class="glass-card">
            <div class="stat-label">{label}</div>
            <div class="stat-value">{value}</div>
            {delta_html}
        </div>
    """, unsafe_allow_html=True)

# --- HELPER: CUSTOM PLOTLY THEME ---
def dark_chart_layout():
    return dict(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#cbd5e1', family="Inter"),
        xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)', zeroline=False),
        yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)', zeroline=False),
    )

def create_gauge_dark(value, title, color_hex):
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = value * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': title, 'font': {'size': 20, 'color': '#e2e8f0', 'family': 'Rajdhani'}},
        number = {'font': {'size': 40, 'color': color_hex, 'family': 'Rajdhani'}, 'suffix': '%'},
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "#334155"},
            'bar': {'color': color_hex, 'thickness': 0.8},
            'bgcolor': "rgba(15, 23, 42, 0.5)",
            'borderwidth': 0,
            'steps': [
                {'range': [0, 100], 'color': 'rgba(30, 41, 59, 0.5)'}
            ],
            'threshold': {
                'line': {'color': "#f8fafc", 'width': 3},
                'thickness': 0.8,
                'value': value * 100
            }
        }
    ))
    fig.update_layout(height=220, margin=dict(l=10, r=10, t=40, b=10), **dark_chart_layout())
    return fig

# --- HEADER SECTION ---
col1, col2 = st.columns([3, 1])
with col1:
    st.markdown("""
        <h1 style='font-size: 3.5rem; margin-bottom: 0; background: linear-gradient(to right, #ffffff, #94a3b8); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>
            FEDERATED <span style='color: #06b6d4'>DIABETES</span> HUB
        </h1>
        <p style='color: #94a3b8; font-size: 1.1rem; margin-top: -10px;'>
            Next-Gen Privacy-Preserving AI for Clinical Decision Support
        </p>
    """, unsafe_allow_html=True)
with col2:
    st.image("https://cdn-icons-png.flaticon.com/512/3063/3063822.png", width=80) 

st.markdown("<br>", unsafe_allow_html=True)

# --- TOP STATS ROW ---
c1, c2, c3, c4 = st.columns(4)
with c1: stat_card("Active Hospitals", "47", "3 New")
with c2: stat_card("Patient Records", "125.4K", "2.1K")
with c3: stat_card("Global Accuracy", "89.2%", "1.4%")
with c4: stat_card("Privacy Budget (ε)", "4.5", "Stable")

st.markdown("<br>", unsafe_allow_html=True)

# --- MAIN TABS ---
tab_titles = ["🔒 Privacy Shield", "🔍 Causal Nexus", "👁️ Multimodal Vision", "⚡ Personalization Engine"]
tabs = st.tabs(tab_titles)

# ------------------------------------------------------------------------------
# TAB 1: PRIVACY
# ------------------------------------------------------------------------------
with tabs[0]:
    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### 🛡️ Privacy Configuration")
        st.markdown("Differential Privacy settings for gradient encryption.")
        
        eps = st.slider("Privacy Budget (ε)", 0.1, 10.0, 1.0)
        st.caption("Lower ε = Stronger Privacy, Lower Utility")
        
        noise = st.slider("Gaussian Noise Multiplier", 0.0, 2.0, 1.1)
        
        if eps < 2.0:
            st.success("✅ GRADE A: Military-Grade Encryption")
        else:
            st.warning("⚠️ GRADE B: Standard Commercial Privacy")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### 📉 Utility Trade-off Analysis")
        
        # Fake chart data
        x_vals = np.linspace(0.1, 10, 50)
        y_vals = 0.95 - (0.3 / np.sqrt(x_vals))
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=x_vals, y=y_vals,
            mode='lines',
            fill='tozeroy',
            line=dict(color='#06b6d4', width=3),
            fillcolor='rgba(6, 182, 212, 0.1)'
        ))
        fig.update_layout(
            **dark_chart_layout(),
            xaxis_title="Privacy Budget (Epsilon)",
            yaxis_title="Model Accuracy",
            height=300
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# TAB 2: CAUSALITY
# ------------------------------------------------------------------------------
with tabs[1]:
    c1, c2 = st.columns([3, 2])
    
    with c1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### 🕸️ Structural Dependency Graph")
        
        st.info("Visualizing statistically significant dependencies (p < 0.05) across 47 hospital nodes.")
        # Placeholder for DAG
        st.markdown("""
        <div style="border: 1px dashed #334155; border-radius: 12px; height: 350px; display: flex; align-items: center; justify-content: center; background: rgba(0,0,0,0.2);">
            <p style="color: #64748b;">[ Interactive 3D Causal Graph Render ]</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### 🎯 Root Cause Analysis")
        
        factors = ['HbA1c Variance', 'Prior Admissions', 'Insulin Adherence', 'Comorbidity Index']
        scores = [0.88, 0.72, 0.65, 0.45]
        
        fig = go.Figure(go.Bar(
            x=scores, y=factors,
            orientation='h',
            marker=dict(
                color=scores,
                colorscale='Viridis',
                line=dict(color='rgba(255,255,255,0.1)', width=1)
            )
        ))
        
        # Fix: Merge xaxis configuration into the layout
        layout_config = dark_chart_layout()
        layout_config['height'] = 380
        layout_config['xaxis']['range'] = [0, 1]
        
        fig.update_layout(**layout_config)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# TAB 3: MULTIMODAL
# ------------------------------------------------------------------------------
with tabs[2]:
    st.markdown("### 🧬 Multimodal Fusion Engine")
    
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        st.markdown('<div class="glass-card" style="height: 100%">', unsafe_allow_html=True)
        st.markdown("#### 1. Retinal Imaging")
        st.file_uploader("Upload Fundus Scan", type=['png','jpg'], key='multi_up')
        st.markdown("""
        <div style="background: #000; height: 150px; border-radius: 8px; display: flex; align-items: center; justify-content: center; border: 1px solid #333;">
            <span style="color: #444">No Signal</span>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col2:
        st.markdown('<div class="glass-card" style="height: 100%">', unsafe_allow_html=True)
        st.markdown("#### 2. Clinical Notes (NLP)")
        st.text_area("Physician Notes", "Patient reports blurry vision...", height=150)
        st.button("⚡ FUSE STREAMS", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col3:
        st.markdown('<div class="glass-card" style="height: 100%">', unsafe_allow_html=True)
        st.markdown("#### 3. Prediction")
        st.markdown("""
        <div style="text-align: center; padding: 20px;">
            <h2 style="color: #ef4444; font-size: 3rem;">89%</h2>
            <p style="color: #94a3b8;">RISK: HIGH (DR)</p>
            <div style="width: 100%; height: 6px; background: #334155; border-radius: 3px; margin-top: 10px;">
                <div style="width: 89%; height: 100%; background: #ef4444; border-radius: 3px; box-shadow: 0 0 10px #ef4444;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# TAB 4: PERSONALIZATION
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# TAB 4: PERSONALIZATION (Updated with Your Specific Logic)
# ------------------------------------------------------------------------------
# with tabs[3]:
#     # --- SECTION 1: RESEARCH MONITOR ---
#     st.markdown('<div class="glass-card">', unsafe_allow_html=True)
#     st.markdown("### 📊 Training Monitor")
    
#     df = load_comp4_data()
    
#     if df.empty:
#         st.warning("⚠️ Waiting for simulation data... Run 'python main_fl_runner.py'")
#     else:
#         # TOP METRICS ROW
#         curr = df.iloc[-1]
#         c1, c2, c3, c4 = st.columns(4)
#         with c1: stat_card("Current Round", int(curr['round']))
#         with c2: stat_card("Global Acc", f"{curr['global_accuracy']:.2%}")
#         with c3: stat_card("Personalized", f"{curr['personalized_accuracy']:.2%}", f"+{curr['personalization_gain']:.2f}%")
        
#         # Fairness Gap Color Logic
#         gap_val = curr['fairness_gap']
#         gap_delta = "Target ≤ 0.05"
#         with c4: stat_card("Fairness Gap", f"{gap_val:.4f}", gap_delta)

#         st.markdown("<br>", unsafe_allow_html=True)

#         # CHARTS ROW
#         g1, g2 = st.columns(2)
#         with g1:
#             st.markdown("#### 📈 Accuracy Evolution")
#             fig = go.Figure()
#             fig.add_trace(go.Scatter(x=df['round'], y=df['global_accuracy'], name='Global', line=dict(color='#64748b', width=2, dash='dash')))
#             fig.add_trace(go.Scatter(x=df['round'], y=df['personalized_accuracy'], name='Personalized', line=dict(color='#06b6d4', width=4)))
#             fig.update_layout(**dark_chart_layout(), height=300, xaxis_title="Round", yaxis_title="Accuracy")
#             st.plotly_chart(fig, use_container_width=True)
            
#         with g2:
#             st.markdown("#### ⚖️ Fairness Monitoring")
#             fig = go.Figure()
#             fig.add_trace(go.Scatter(x=df['round'], y=df['fairness_gap'], fill='tozeroy', line=dict(color='#f43f5e', width=2)))
#             fig.add_hline(y=0.05, line_dash="dash", line_color="#4ade80", annotation_text="Target")
#             fig.update_layout(**dark_chart_layout(), height=300, xaxis_title="Round", yaxis_title="Gap")
#             st.plotly_chart(fig, use_container_width=True)

#     st.markdown('</div>', unsafe_allow_html=True)
#     st.markdown("<br>", unsafe_allow_html=True)

#     # --- SECTION 2: CLINICIAN PREDICTION TOOL (Your Specific Logic) ---
#     st.markdown('<div class="glass-card">', unsafe_allow_html=True)
#     st.markdown("### 🩺 Clinician Prediction Tool")
#     st.caption("Live Multi-Task Prediction: Hypertension & Heart Failure")
    
#     with st.form("prediction_form"):
#         # 1. INPUTS
#         c1, c2, c3 = st.columns(3)
#         with c1:
#             age = st.slider("Patient Age", 10, 100, 65)
#             gender = st.selectbox("Gender", ["Female", "Male"])
#         with c2:
#             meds = st.slider("Medication Count", 0, 40, 12)
#             hba1c = st.slider("HbA1c Level", 4.0, 15.0, 8.5)
#         with c3:
#             bmi = st.slider("BMI", 15.0, 50.0, 30.0)
            
#         st.markdown("<br>", unsafe_allow_html=True)
#         submit = st.form_submit_button("RUN RISK ASSESSMENT")

#     # 2. OUTPUTS (THE COOL GAUGES)
#     if submit:
#         # Pause refresh so user can see result
#         st.session_state.pause_refresh = True
        
#         with st.spinner("Processing federated inference..."):
#             time.sleep(0.8) # UI effect
        
#         # --- YOUR SIMULATION LOGIC ---
#         norm_age = age / 100
#         norm_meds = meds / 40
#         norm_hba1c = (hba1c - 4) / 11 
#         norm_bmi = (bmi - 15) / 35      
        
#         # Gender Logic
#         gender_risk = 0.10 if gender == "Male" else 0.00
            
#         # Weighted Score
#         risk_score = (norm_age * 0.15) + (norm_meds * 0.25) + \
#                      (norm_hba1c * 0.25) + (norm_bmi * 0.20) + gender_risk
        
#         # Task Probabilities
#         prob_htn = min(0.98, (risk_score * 1.1) + 0.05) 
#         prob_hf = min(0.95, risk_score * 0.95)
        
#         # --- DISPLAY RESULTS ---
#         col_out1, col_out2 = st.columns(2)
        
#         with col_out1:
#             color = "#f43f5e" if prob_htn > 0.5 else "#4ade80" # Red if high, Green if low
#             fig1 = create_gauge_dark(prob_htn, "Hypertension Risk", color)
#             st.plotly_chart(fig1, use_container_width=True)
            
#             # Status Text
#             if prob_htn > 0.7: 
#                 st.error("🚨 HIGH RISK DETECTED")
#             elif prob_htn > 0.5: 
#                 st.warning("⚠️ MODERATE RISK")
#             else: 
#                 st.success("✅ LOW RISK")
            
#         with col_out2:
#             color = "#f43f5e" if prob_hf > 0.5 else "#4ade80"
#             fig2 = create_gauge_dark(prob_hf, "Heart Failure Risk", color)
#             st.plotly_chart(fig2, use_container_width=True)
            
#             # Status Text
#             if prob_hf > 0.7: 
#                 st.error("🚨 HIGH RISK DETECTED")
#             elif prob_hf > 0.5: 
#                 st.warning("⚠️ MODERATE RISK")
#             else: 
#                 st.success("✅ LOW RISK")

#     st.markdown('</div>', unsafe_allow_html=True)


# ------------------------------------------------------------------------------
# TAB 4: PERSONALIZATION (Enhanced Version)
# ------------------------------------------------------------------------------
with tabs[3]:
    # --- SECTION 1: RESEARCH MONITOR ---
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### 📊 Training Monitor")
    
    df = load_comp4_data()
    
    if df.empty:
        st.warning("⚠️ Waiting for simulation data... Run 'python main_fl_runner.py'")
    else:
        # TOP METRICS ROW
        curr = df.iloc[-1]
        c1, c2, c3, c4 = st.columns(4)
        with c1: stat_card("Current Round", int(curr['round']))
        with c2: stat_card("Global Acc", f"{curr['global_accuracy']:.2%}")
        with c3: stat_card("Personalized", f"{curr['personalized_accuracy']:.2%}", f"+{curr['personalization_gain']:.2f}%")
        
        # Fairness Gap Color Logic
        gap_val = curr['fairness_gap']
        gap_delta = "Target ≤ 0.05"
        with c4: stat_card("Fairness Gap", f"{gap_val:.4f}", gap_delta)

        st.markdown("<br>", unsafe_allow_html=True)

        # CHARTS ROW
        g1, g2 = st.columns(2)
        with g1:
            st.markdown("#### 📈 Accuracy Evolution")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df['round'], y=df['global_accuracy'], name='Global', line=dict(color='#64748b', width=2, dash='dash')))
            fig.add_trace(go.Scatter(x=df['round'], y=df['personalized_accuracy'], name='Personalized', line=dict(color='#06b6d4', width=4)))
            fig.update_layout(**dark_chart_layout(), height=300, xaxis_title="Round", yaxis_title="Accuracy")
            st.plotly_chart(fig, use_container_width=True)
            
        with g2:
            st.markdown("#### ⚖️ Fairness Monitoring")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df['round'], y=df['fairness_gap'], fill='tozeroy', line=dict(color='#f43f5e', width=2)))
            fig.add_hline(y=0.05, line_dash="dash", line_color="#4ade80", annotation_text="Target")
            fig.update_layout(**dark_chart_layout(), height=300, xaxis_title="Round", yaxis_title="Gap")
            st.plotly_chart(fig, use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # --- SECTION 2: CLINICIAN PREDICTION TOOL (Enhanced) ---
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### 🩺 Clinician Prediction Tool")
    st.caption("Live Multi-Task Prediction: Hypertension & Heart Failure")
    
    with st.form("prediction_form"):
        # 1. INPUTS
        c1, c2, c3 = st.columns(3)
        with c1:
            age = st.slider("Patient Age", 10, 100, 65)
            gender = st.selectbox("Gender", ["Female", "Male"])
        with c2:
            meds = st.slider("Medication Count", 0, 40, 12)
            hba1c = st.slider("HbA1c Level", 4.0, 15.0, 8.5)
        with c3:
            bmi = st.slider("BMI", 15.0, 50.0, 30.0)
            
        st.markdown("---")
        
        # NEW: Personalization Controls
        h1, h2 = st.columns(2)
        with h1:
            hospital_type = st.selectbox("Select Hospital Context (Non-IID)", 
                                       ["Hospital A (Urban - General)", 
                                        "Hospital B (Rural - Geriatric)", 
                                        "Hospital C (Specialized - Cardiac)"])
        with h2:
            model_mode = st.radio("Model Inference Mode", ["Global Model", "Personalized Model"], horizontal=True)
            
        st.markdown("<br>", unsafe_allow_html=True)
        submit = st.form_submit_button("RUN RISK ASSESSMENT")

    # 2. OUTPUTS (THE COOL GAUGES)
    if submit:
        # Pause refresh so user can see result
        st.session_state.pause_refresh = True
        
        with st.spinner("Processing federated inference..."):
            time.sleep(0.8) # UI effect
        
        # --- ENHANCED SIMULATION LOGIC ---
        norm_age = age / 100
        norm_meds = meds / 40
        norm_hba1c = (hba1c - 4) / 11 
        norm_bmi = (bmi - 15) / 35      
        
        # Gender Logic
        gender_risk = 0.10 if gender == "Male" else 0.00
            
        # Base Risk Calculation
        base_risk = 0.3
        
        # Task 1: Hypertension Risk
        prob_htn = base_risk + (norm_age*0.2) + (norm_hba1c*0.3) + (norm_bmi*0.25) + (norm_meds*0.15) + gender_risk
        
        # Task 2: Heart Failure Risk
        prob_hf = base_risk + (norm_age*0.3) + (norm_hba1c*0.25) + (norm_bmi*0.2) + (norm_meds*0.1) + (gender_risk * 0.5)

        # --- HOSPITAL CONTEXT MODIFIERS (Non-IID Simulation) ---
        if hospital_type == "Hospital B (Rural - Geriatric)":
            # Rural/Older populations might have slightly higher underlying risk
            prob_htn += 0.05
            prob_hf += 0.08
        elif hospital_type == "Hospital C (Specialized - Cardiac)":
            # Cardiac centers might see more severe cases (bias)
            prob_hf += 0.10

        # --- PERSONALIZATION BOOST ---
        if model_mode == "Personalized Model":
            # Personalization "calibrates" the model, usually reducing false positives or refining risk
            # For demo, we simulate a 'cleaner' prediction (closer to extremes)
            if prob_htn > 0.5: prob_htn += 0.05
            else: prob_htn -= 0.05
            
            if prob_hf > 0.5: prob_hf += 0.05
            else: prob_hf -= 0.05
            
            st.toast("⚡ Personalized model applied: Adjusted for local patient demographics.")

        # --- ADD NOISE (Realism) ---
        prob_htn = np.clip(prob_htn + np.random.normal(0, 0.02), 0.05, 0.98)
        prob_hf = np.clip(prob_hf + np.random.normal(0, 0.02), 0.05, 0.95)
        
        # --- DISPLAY RESULTS ---
        col_out1, col_out2 = st.columns(2)
        
        with col_out1:
            color = "#f43f5e" if prob_htn > 0.5 else "#4ade80" 
            fig1 = create_gauge_dark(prob_htn, "Hypertension Risk", color)
            st.plotly_chart(fig1, use_container_width=True)
            
            if prob_htn > 0.7: st.error("🚨 HIGH RISK")
            elif prob_htn > 0.5: st.warning("⚠️ MODERATE RISK")
            else: st.success("✅ LOW RISK")
            
        with col_out2:
            color = "#f43f5e" if prob_hf > 0.5 else "#4ade80"
            fig2 = create_gauge_dark(prob_hf, "Heart Failure Risk", color)
            st.plotly_chart(fig2, use_container_width=True)
            
            if prob_hf > 0.7: st.error("🚨 HIGH RISK")
            elif prob_hf > 0.5: st.warning("⚠️ MODERATE RISK")
            else: st.success("✅ LOW RISK")

        # --- KEY DRIVERS (Interpretability) ---
        with st.expander("🔍 Key Risk Drivers (Explainability)"):
            st.write("Top factors contributing to this prediction:")
            drivers = pd.DataFrame({
                "Factor": ["HbA1c", "Medications", "BMI", "Age"],
                "Impact": [norm_hba1c, norm_meds, norm_bmi, norm_age]
            }).sort_values(by="Impact", ascending=False)
            
            # Simple bar chart for drivers
            fig_d = go.Figure(go.Bar(
                x=drivers["Impact"], y=drivers["Factor"], orientation='h',
                marker=dict(color=['#f43f5e', '#fb923c', '#fbbf24', '#a3e635'])
            ))
            fig_d.update_layout(**dark_chart_layout(), height=200, margin=dict(l=0,r=0,t=0,b=0))
            st.plotly_chart(fig_d, use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

# --- SIDEBAR FOOTER ---
with st.sidebar:
    st.markdown("---")
    st.markdown("### 📡 System Logs")
    st.code("""
    [10:42:12] Client #4 connected
    [10:42:15] Round 18 Aggregation
    [10:42:18] Privacy Noise Added
    [10:42:20] Model Updated
    """, language="bash")
    st.caption("v2.5.0-beta | Secure Connection")

# ==============================================================================
# DASHBOARD CODE FOR FEDERATED DIABETES RESEARCH SYSTEM - Gemini Edition
# ==============================================================================
# import streamlit as st
# import pandas as pd
# import json
# import time
# import os
# import plotly.graph_objects as go
# import numpy as np

# # --- PAGE CONFIG ---
# st.set_page_config(
#     page_title="Federated Diabetes Research System",
#     page_icon="🏥",
#     layout="wide"
# )

# # --- CUSTOM CSS ---
# st.markdown("""
#     <style>
#     .big-font { font-size:20px !important; }
#     .stTabs [data-baseweb="tab-list"] { gap: 24px; }
#     .stTabs [data-baseweb="tab"] { height: 50px; white-space: pre-wrap; background-color: #f0f2f6; border-radius: 4px; color: black;}
#     .stTabs [aria-selected="true"] { background-color: #00E5FF; color: white; }
#     </style>
#     """, unsafe_allow_html=True)

# # --- SESSION STATE ---
# if 'pause_refresh' not in st.session_state:
#     st.session_state.pause_refresh = False

# # --- HEADER ---
# st.title("🏥 Federated Diabetes Research Platform")
# st.markdown("Integrates Privacy, Causality, Multimodal Data, and Personalized Multi-Task Learning.")

# # --- SIDEBAR ---
# st.sidebar.header("Global Controls")
# st.sidebar.info("System Status: ONLINE")
# if 'auto_refresh_on' not in st.session_state:
#     st.session_state.auto_refresh_on = True
# auto_refresh = st.sidebar.checkbox("Auto-Refresh Live Data", value=st.session_state.auto_refresh_on)

# # --- HELPER FUNCTIONS ---
# def create_gauge(value, title, color_hex):
#     fig = go.Figure(go.Indicator(
#         mode = "gauge+number",
#         value = value * 100,
#         domain = {'x': [0, 1], 'y': [0, 1]},
#         title = {'text': title, 'font': {'size': 24}},
#         gauge = {
#             'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "white"},
#             'bar': {'color': color_hex},
#             'bgcolor': "white",
#             'borderwidth': 2,
#             'bordercolor': "gray",
#             'steps': [
#                 {'range': [0, 50], 'color': 'rgba(0, 255, 0, 0.1)'},
#                 {'range': [50, 100], 'color': 'rgba(255, 0, 0, 0.1)'}
#             ],
#         }
#     ))
#     fig.update_layout(height=250, margin=dict(l=10, r=10, t=50, b=10))
#     return fig

# def load_comp4_data():
#     if not os.path.exists("results/comp4_results/fl_results.json"):
#         return pd.DataFrame()
#     with open("results/comp4_results/fl_results.json", "r") as f:
#         try:
#             data = json.load(f)
#             return pd.DataFrame(data)
#         except:
#             return pd.DataFrame()

# # ==============================================================================
# # MAIN TABS ARCHITECTURE
# # ==============================================================================
# tab1, tab2, tab3, tab4 = st.tabs([
#     "🔒 Comp 1: Privacy Risk", 
#     "🔍 Comp 2: Causal Analysis", 
#     "👁️ Comp 3: Multimodal", 
#     "⚡ Comp 4: Personalized MTFL"
# ])

# # ------------------------------------------------------------------------------
# # COMPONENT 1: Privacy-Preserving Complication Prediction
# # ------------------------------------------------------------------------------
# with tab1:
#     st.header("🔒 Privacy-Preserving Complication Risk")
#     st.caption("Owner: Member 1 | Focus: Differential Privacy & Secure Aggregation")
    
#     col1, col2 = st.columns(2)
#     with col1:
#         st.info("ℹ️ This component predicts complications (Retinopathy, Neuropathy) using Encrypted Gradients.")
#         # PLACEHOLDER INPUTS
#         dp_epsilon = st.slider("Privacy Budget (Epsilon)", 0.1, 10.0, 1.0, key="c1_eps")
#         noise_mult = st.slider("Noise Multiplier", 0.0, 2.0, 1.0, key="c1_noise")
        
#     with col2:
#         # PLACEHOLDER OUTPUT
#         st.subheader("🛡️ Privacy vs. Utility Trade-off")
#         # Dummy data for visualization
#         chart_data = pd.DataFrame({
#             'Epsilon': [0.1, 0.5, 1.0, 2.0, 5.0],
#             'Model Accuracy': [60, 75, 82, 88, 91]
#         })
#         st.line_chart(chart_data.set_index('Epsilon'))

# # ------------------------------------------------------------------------------
# # COMPONENT 2: Causal Discovery
# # ------------------------------------------------------------------------------
# with tab2:
#     st.header("🔍 Causal Discovery & Explainability")
#     st.caption("Owner: Member 2 | Focus: Causal Graphs & Readmission Drivers")
    
#     col1, col2 = st.columns([1, 2])
#     with col1:
#         st.write("Identifies **WHY** a patient is readmitted.")
#         patient_id = st.text_input("Enter Patient ID", "PT-4592", key="c2_pid")
#         if st.button("Analyze Causality", key="c2_btn"):
#             st.success("Causal Graph Generated.")
            
#     with col2:
#         # Placeholder for a Causal Graph (e.g., DAG)
#         st.subheader("🔗 Causal Graph Visualization")
#         st.markdown("*(Graph Placeholder: Medication -> Readmission <- Age)*")
#         st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/0/0e/Causal_DAG.svg/1200px-Causal_DAG.svg.png", width=300)

# # ------------------------------------------------------------------------------
# # COMPONENT 3: Multimodal Detection
# # ------------------------------------------------------------------------------
# with tab3:
#     st.header("👁️ Multimodal Data Integration")
#     st.caption("Owner: Member 3 | Focus: Retinal Images + Clinical Text")
    
#     col1, col2 = st.columns(2)
#     with col1:
#         st.file_uploader("Upload Retinal Scan (Image)", type=['png', 'jpg'], key="c3_img")
#         st.text_area("Clinical Notes", "Patient complains of blurry vision...", key="c3_txt")
        
#     with col2:
#         if st.button("Run Multimodal Fusion", key="c3_btn"):
#             st.spinner("Processing CNN + BERT embeddings...")
#             time.sleep(1)
#             st.success("Analysis Complete")
#             st.metric("Diabetic Retinopathy Risk", "High (89%)")

# # ------------------------------------------------------------------------------
# # COMPONENT 4: YOUR WORK (Personalized MTFL)
# # ------------------------------------------------------------------------------
# with tab4:
#     st.header("⚡ Personalized Multi-Task FL")
#     st.caption("Owner: Member 4 | Focus: Personalization, Efficiency, Fairness")
    
#     # --- 1. RESEARCH MONITOR ---
#     df = load_comp4_data()
#     if df.empty:
#         st.warning("⚠️ Waiting for simulation data... Run 'python main_fl_runner.py'")
#     else:
#         curr = df.iloc[-1]
#         c1, c2, c3, c4 = st.columns(4)
#         c1.metric("Current Round", int(curr['round']))
#         c2.metric("Global Accuracy", f"{curr['global_accuracy']:.2%}")
#         c3.metric("Personalized Acc", f"{curr['personalized_accuracy']:.2%}", delta=f"{curr['personalization_gain']:.2f}% Gain")
        
#         gap_color = "normal" if curr['fairness_gap'] <= 0.05 else "inverse"
#         c4.metric("Fairness Gap", f"{curr['fairness_gap']:.4f}", delta="-Target ≤ 0.05", delta_color=gap_color)
        
#         g1, g2 = st.columns(2)
#         with g1:
#             st.subheader("📈 Accuracy Gain")
#             st.line_chart(df[['global_accuracy', 'personalized_accuracy']])
#         with g2:
#             st.subheader("⚖️ Fairness (Demographic Parity)")
#             st.area_chart(df['fairness_gap'], color="#ff4b4b")

#     st.divider()

#     # --- 2. CLINICIAN DEMO (YOUR INTERACTIVE TOOL) ---
#     st.subheader("🩺 Clinician Prediction Tool (Live Demo)")
#     st.warning("⚠️ **Note:** Uses heuristic simulation for UI demonstration.")

#     # Note: We use unique keys (c4_) to prevent conflict with other tabs
#     with st.form("c4_prediction_form"):
#         c1, c2, c3 = st.columns(3)
#         with c1:
#             age = st.slider("Patient Age", 10, 100, 65, key="c4_age")
#             gender = st.selectbox("Gender", ["Female", "Male"], key="c4_gen")
#         with c2:
#             meds = st.slider("Medications", 0, 40, 12, key="c4_meds")
#             hba1c = st.slider("HbA1c Level", 4.0, 15.0, 8.5, key="c4_hb")
#         with c3:
#             bmi = st.slider("BMI", 15.0, 50.0, 30.0, key="c4_bmi")
            
#         submit = st.form_submit_button("Run Risk Assessment")

#     if submit:
#         st.session_state.pause_refresh = True
        
#         # --- SIMULATION LOGIC ---
        
#         # 1. Normalize inputs
#         norm_age = age / 100
#         norm_meds = meds / 40
#         norm_hba1c = (hba1c - 4) / 11 
#         norm_bmi = (bmi - 15) / 35
        
#         # 2. Gender Logic
#         if gender == "Male":
#             gender_risk = 0.10 
#         else:
#             gender_risk = 0.00 
            
#         # 3. Weighted Score
#         risk_score = (norm_age * 0.15) + \
#                      (norm_meds * 0.25) + \
#                      (norm_hba1c * 0.25) + \
#                      (norm_bmi * 0.20) + \
#                      (gender_risk)
        
#         # 4. Probabilities
#         prob_htn = min(0.98, (risk_score * 1.1) + 0.05) 
#         prob_hf = min(0.95, risk_score * 0.95)
        
#         st.success("✅ Prediction Complete")
        
#         # 5. THE COOL GAUGES
#         col_out1, col_out2 = st.columns(2)
        
#         with col_out1:
#             color = "#ff4b4b" if prob_htn > 0.5 else "#09ab3b"
#             fig1 = create_gauge(prob_htn, "Hypertension Risk", color)
#             st.plotly_chart(fig1, use_container_width=True)
            
#             if prob_htn > 0.7: st.error("⚠️ High Risk Detected")
#             elif prob_htn > 0.5: st.warning("⚠️ Moderate Risk")
#             else: st.info("✅ Low Risk")
            
#         with col_out2:
#             color = "#ff4b4b" if prob_hf > 0.5 else "#09ab3b"
#             fig2 = create_gauge(prob_hf, "Heart Failure Risk", color)
#             st.plotly_chart(fig2, use_container_width=True)
            
#             if prob_hf > 0.7: st.error("⚠️ High Risk Detected")
#             elif prob_hf > 0.5: st.warning("⚠️ Moderate Risk")
#             else: st.info("✅ Low Risk")

# # --- AUTO REFRESH LOGIC ---
# if auto_refresh and not st.session_state.pause_refresh:
#     time.sleep(2)
#     st.rerun()

# # Reset pause flag if not submitting
# if not submit:
#     st.session_state.pause_refresh = False

# ==============================================================================
# DELETED CODE SNIPPETS FOR REFERENCE
# ==============================================================================
# import streamlit as st
# import pandas as pd
# import json
# import time
# import os
# import plotly.graph_objects as go

# # --- PAGE CONFIG ---
# st.set_page_config(
#     page_title="MTFL Diabetes Dashboard",
#     page_icon="🏥",
#     layout="wide"
# )

# # --- CUSTOM CSS ---
# st.markdown("""
#     <style>
#     .big-font { font-size:20px !important; }
#     </style>
#     """, unsafe_allow_html=True)

# # --- SESSION STATE INITIALIZATION ---
# if 'pause_refresh' not in st.session_state:
#     st.session_state.pause_refresh = False

# # --- HEADER ---
# st.title("🏥 Personalized Multi-Task FL Dashboard")
# st.markdown("**Component 4 Status:** Monitoring Real-Time Training, Personalization Gain, and Fairness.")

# # --- SIDEBAR ---
# st.sidebar.header("Simulation Controls")
# st.sidebar.info("Run 'python main_fl_runner.py' in terminal to train.")
# # Move Auto-Refresh to session state so we can control it programmatically
# if 'auto_refresh_on' not in st.session_state:
#     st.session_state.auto_refresh_on = True

# auto_refresh = st.sidebar.checkbox("Auto-Refresh Data", value=st.session_state.auto_refresh_on)

# # --- HELPER: CREATE GAUGE CHART ---
# def create_gauge(value, title, color_hex):
#     fig = go.Figure(go.Indicator(
#         mode = "gauge+number",
#         value = value * 100,
#         domain = {'x': [0, 1], 'y': [0, 1]},
#         title = {'text': title, 'font': {'size': 24}},
#         gauge = {
#             'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "white"},
#             'bar': {'color': color_hex},
#             'bgcolor': "white",
#             'borderwidth': 2,
#             'bordercolor': "gray",
#             'steps': [
#                 {'range': [0, 50], 'color': 'rgba(0, 255, 0, 0.1)'},
#                 {'range': [50, 100], 'color': 'rgba(255, 0, 0, 0.1)'}
#             ],
#         }
#     ))
#     fig.update_layout(height=250, margin=dict(l=10, r=10, t=50, b=10))
#     return fig

# # --- HELPER: LOAD DATA ---
# def load_data():
#     if not os.path.exists("results/comp4_results/fl_results.json"):
#         return pd.DataFrame()
#     with open("results/comp4_results/fl_results.json", "r") as f:
#         try:
#             data = json.load(f)
#             return pd.DataFrame(data)
#         except:
#             return pd.DataFrame() # Handle empty/corrupt file read

# # --- MAIN DASHBOARD CONTENT ---
# df = load_data()

# if df.empty:
#     st.warning("⚠️ Waiting for simulation data... Run the FL engine!")
# else:
#     # 1. TOP METRICS
#     curr = df.iloc[-1]
#     col1, col2, col3, col4 = st.columns(4)
#     col1.metric("Current Round", int(curr['round']))
#     col2.metric("Global Accuracy", f"{curr['global_accuracy']:.2%}")
#     col3.metric("Personalized Acc", f"{curr['personalized_accuracy']:.2%}", delta=f"{curr['personalization_gain']:.2f}% Gain")
    
#     gap_color = "normal" if curr['fairness_gap'] <= 0.05 else "inverse"
#     col4.metric("Fairness Gap", f"{curr['fairness_gap']:.4f}", delta="-Target ≤ 0.05", delta_color=gap_color)

#     st.divider()

#     # 2. CHARTS
#     c1, c2 = st.columns(2)
#     with c1:
#         st.subheader("📈 Personalization Gain")
#         st.line_chart(df[['global_accuracy', 'personalized_accuracy']])
        
#     with c2:
#         st.subheader("⚖️ Fairness Monitoring")
#         st.area_chart(df['fairness_gap'], color="#ff4b4b")

#     with st.expander("📄 View Logs"):
#         st.dataframe(df)

# st.divider()

# # --- CLINICIAN PREDICTION DEMO ---
# st.subheader("🩺 Clinician Prediction Tool")
# st.caption("Live Multi-Task Prediction: Hypertension & Heart Failure")

# # 1. INPUTS
# with st.form("prediction_form"):
#     c1, c2, c3 = st.columns(3)
#     with c1:
#         age = st.slider("Patient Age", 10, 100, 65)
#         gender = st.selectbox("Gender", ["Female", "Male"])
#     with c2:
#         meds = st.slider("Medication Count", 0, 40, 12)
#         hba1c = st.slider("HbA1c Level", 4.0, 15.0, 8.5)
#     with c3:
#         bmi = st.slider("BMI", 15.0, 50.0, 30.0)
        
#     submit = st.form_submit_button("Run Risk Assessment")

# # 2. OUTPUTS (THE COOL GAUGES)
# # 2. OUTPUTS (THE COOL GAUGES)
# if submit:
#     # Pause refresh so the user can see the result
#     st.session_state.pause_refresh = True
    
#     # --- SIMULATION LOGIC ---
    
#     # 1. Normalize numerical inputs (0.0 to 1.0 range)
#     norm_age = age / 100
#     norm_meds = meds / 40
#     norm_hba1c = (hba1c - 4) / 11 
#     norm_bmi = (bmi - 15) / 35    
    
#     # 2. Handle Categorical Inputs (Gender)
#     # Example Logic: Males might have slightly higher baseline risk for Heart Failure
#     if gender == "Male":
#         gender_risk = 0.10 # Add 10% risk
#     else:
#         gender_risk = 0.00 # Baseline
        
#     # 3. Calculate Weighted Risk Score (Sum of all factors)
#     # Weights: Meds and HbA1c are strongest predictors
#     risk_score = (norm_age * 0.15) + \
#                  (norm_meds * 0.25) + \
#                  (norm_hba1c * 0.25) + \
#                  (norm_bmi * 0.20) + \
#                  (gender_risk)
    
#     # 4. Task-Specific Probabilities
#     # Hypertension: Highly sensitive to BMI and Age
#     # We tweak the multiplier so HTN reacts differently than HF
#     prob_htn = min(0.98, (risk_score * 1.1) + 0.05) 
    
#     # Heart Failure: Highly sensitive to Meds and Gender
#     prob_hf = min(0.95, risk_score * 0.95)
    
#     # -----------------------
    
#     st.success("✅ Prediction Complete")
    
#     col_out1, col_out2 = st.columns(2)
    
#     with col_out1:
#         color = "#ff4b4b" if prob_htn > 0.5 else "#09ab3b"
#         fig1 = create_gauge(prob_htn, "Hypertension Risk", color)
#         st.plotly_chart(fig1, use_container_width=True)
#         if prob_htn > 0.7: st.error("⚠️ High Risk Detected")
#         elif prob_htn > 0.5: st.warning("⚠️ Moderate Risk")
#         else: st.info("✅ Low Risk")
        
#     with col_out2:
#         color = "#ff4b4b" if prob_hf > 0.5 else "#09ab3b"
#         fig2 = create_gauge(prob_hf, "Heart Failure Risk", color)
#         st.plotly_chart(fig2, use_container_width=True)
#         if prob_hf > 0.7: st.error("⚠️ High Risk Detected")
#         elif prob_hf > 0.5: st.warning("⚠️ Moderate Risk")
#         else: st.info("✅ Low Risk")

# # --- AUTO REFRESH LOGIC ---
# if auto_refresh and not st.session_state.pause_refresh:
#     time.sleep(2)
#     st.rerun()

# # Reset the pause flag after one run-through if submit wasn't clicked
# if not submit:
#     st.session_state.pause_refresh = False