import streamlit as st
import requests
import numpy as np
import plotly.graph_objects as go
import time

# --- Config & State ---
API_URL = "http://api:8000/predict"
st.set_page_config(page_title="Project Ascension | L3 Radar", layout="wide", initial_sidebar_state="collapsed")

# Custom CSS for Quant Terminal look
st.markdown("""
    <style>
    .stApp { background-color: #0E1117; color: #C9D1D9; font-family: 'Courier New', Courier, monospace; }
    .metric-box { background-color: #161B22; border: 1px solid #30363D; border-radius: 5px; padding: 15px; text-align: center; }
    .critical { border-left: 5px solid #FF4444; }
    .stable { border-left: 5px solid #00C851; }
    </style>
""", unsafe_allow_html=True)

st.title("MARKET MICROSTRUCTURE RADAR")
st.markdown("---")

# --- Layout ---
top_col1, top_col2, top_col3 = st.columns(3)
main_col, side_col = st.columns([3, 1])

with side_col:
    st.markdown("### 🎛️ Execution Control")
    if st.button("PULSE L3 FEED", use_container_width=True, type="primary"):
        with st.spinner("Intercepting Tick Data..."):
            # Synthetic 128-tick burst with a heavy liquidation drop
            ticks = np.random.randint(40, 60, 128)
            ticks[90:120] = np.random.randint(10, 25, 30) # Floor falls out
            
            start_time = time.time()
            try:
                response = requests.post(API_URL, json={"ticks": ticks.tolist()})
                if response.status_code == 200:
                    st.session_state['data'] = response.json()
                    st.session_state['ticks'] = ticks
                    st.session_state['rtt'] = (time.time() - start_time) * 1000
                else:
                    st.error(f"API Reject: {response.status_code}")
            except Exception as e:
                st.error("Engine Disconnected.")

with main_col:
    st.markdown("### 📊 LOB Tick Trajectory (Window: 128ms)")
    if 'ticks' in st.session_state:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=st.session_state['ticks'], 
            mode='lines+markers', 
            line=dict(color='#00FFCC', width=2),
            marker=dict(size=4, color='#FF4444'),
            fill='tozeroy',
            fillcolor='rgba(0, 255, 204, 0.1)'
        ))
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            height=450, margin=dict(l=0, r=0, t=30, b=0),
            xaxis=dict(showgrid=False, zeroline=False, color='#8B949E'),
            yaxis=dict(showgrid=True, gridcolor='#30363D', color='#8B949E')
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Awaiting execution pulse...")

# --- Telemetry ---
if 'data' in st.session_state:
    data = st.session_state['data']
    prob = data['probability'] * 100
    is_imminent = data['cascade_imminent']
    
    with top_col1:
        st.markdown(f"<div class='metric-box {'critical' if is_imminent else 'stable'}'>"
                    f"<h4 style='margin:0; color:#8B949E;'>CASCADE PROBABILITY</h4>"
                    f"<h1 style='margin:0; color:{'#FF4444' if is_imminent else '#00C851'};'>{prob:.2f}%</h1>"
                    f"</div>", unsafe_allow_html=True)
    
    with top_col2:
        st.markdown(f"<div class='metric-box'>"
                    f"<h4 style='margin:0; color:#8B949E;'>MODEL LATENCY</h4>"
                    f"<h1 style='margin:0; color:#58A6FF;'>{data['latency_ms']:.2f} ms</h1>"
                    f"</div>", unsafe_allow_html=True)
        
    with top_col3:
        st.markdown(f"<div class='metric-box'>"
                    f"<h4 style='margin:0; color:#8B949E;'>NETWORK RTT</h4>"
                    f"<h1 style='margin:0; color:#D2A8FF;'>{st.session_state['rtt']:.2f} ms</h1>"
                    f"</div>", unsafe_allow_html=True)
