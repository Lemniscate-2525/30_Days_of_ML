import streamlit as st
import requests
import numpy as np
import plotly.graph_objects as go

# Connects to the FastAPI Docker container
API_URL = "http://api:8000/predict"

st.set_page_config(page_title="HFT Risk Desk", layout="wide", page_icon="📉")
st.title("Project Ascension: Microstructure Risk Radar")

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Live Limit Order Book (LOB) State")
    if st.button("Simulate Incoming Market Trajectory"):
        with st.spinner("Intercepting L3 Feed..."):
            simulated_ticks = np.random.randint(0, 100, 128)
            simulated_ticks[80:110] = np.random.randint(90, 100, 30) # Crash signature
            
            fig = go.Figure(data=go.Scatter(y=simulated_ticks, mode='lines', line=dict(color='cyan')))
            fig.update_layout(template="plotly_dark", height=400, title="Micro-Tick Volatility (128 units)")
            st.plotly_chart(fig, use_container_width=True)
            
            try:
                response = requests.post(API_URL, json={"ticks": simulated_ticks.tolist()})
                if response.status_code == 200:
                    st.session_state['pred'] = response.json()
                else:
                    st.error("API Connection Error")
            except Exception as e:
                st.error(f"Failed to reach API: {e}")

with col2:
    st.subheader("Llama-3 Threat Assessment")
    if 'pred' in st.session_state:
        data = st.session_state['pred']
        prob = data['probability'] * 100
        
        color = "red" if data['cascade_imminent'] else "green"
        st.markdown(f"<h1 style='text-align: center; color: {color};'>{prob:.2f}%</h1>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center;'>Probability of Liquidation Cascade</p>", unsafe_allow_html=True)
        
        if data['cascade_imminent']:
            st.error("🚨 CRITICAL WARNING: LIQUIDITY VACUUM DETECTED. HALT EXECUTION.")
        else:
            st.success("✅ Market Microstructure Stable.")
            
        st.metric(label="Inference Engine Latency", value=f"{data['latency_ms']:.2f} ms")
