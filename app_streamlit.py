import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go

# Set page config
st.set_page_config(
    page_title="Energy Efficiency Predictor",
    page_icon="⚡",
    layout="wide"
)

# Add custom CSS
st.markdown("""
    <style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .prediction-box {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .metric-box {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 5px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 10px 0;
    }
    .metric-title {
        color: #000000 !important;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# Title and description
st.title("Energy Efficiency Predictor")
st.markdown("""
    This app predicts CO2 emissions and energy efficiency for carbon capture processes 
    based on various input parameters. Adjust the parameters below to see the predictions.
""")

# Load model and preprocessors
@st.cache_resource
def load_model():
    model_path = Path("model_files")
    model = joblib.load(model_path / "carbon_emission_model.joblib")
    label_encoder = joblib.load(model_path / "label_encoder.joblib")
    imputer = joblib.load(model_path / "imputer.joblib")
    scaler = joblib.load(model_path / "scaler.joblib")
    
    # Get feature names from the model
    if hasattr(model, 'feature_names_in_'):
        feature_names = model.feature_names_in_
    else:
        # If using MultiOutputRegressor, get feature names from the first estimator
        feature_names = model.estimators_[0].feature_names_in_
    
    return model, label_encoder, imputer, scaler, feature_names

try:
    model, label_encoder, imputer, scaler, model_feature_names = load_model()
    st.sidebar.write("Model feature names:", model_feature_names)
except Exception as e:
    st.error(f"Error loading model: {str(e)}")
    st.stop()

# Create two columns for input parameters
col1, col2 = st.columns(2)

with col1:
    st.subheader("Primary Parameters")
    temperature = st.number_input("Temperature (°C)", value=85.0, min_value=0.0, max_value=200.0)
    pressure = st.number_input("Pressure (bar)", value=5.0, min_value=0.1, max_value=50.0)
    co2_concentration = st.number_input("CO2 Concentration (%)", value=12.0, min_value=0.0, max_value=100.0)
    flow_rate = st.number_input("Flow Rate (m³/h)", value=1000.0, min_value=0.0)
    solvent_concentration = st.number_input("Solvent Concentration (%)", value=30.0, min_value=0.0, max_value=100.0)
    sorbent_type = st.selectbox(
        "Sorbent Type",
        options=["Activated Carbon", "Zeolite", "MOF", "Amine"]
    )

with col2:
    st.subheader("Secondary Parameters")
    capture_efficiency = st.number_input("Capture Efficiency (%)", value=85.0, min_value=0.0, max_value=100.0)
    energy_consumption = st.number_input("Energy Consumption (MJ/kg CO2)", value=2.5, min_value=0.0)
    co2_outlet = st.number_input("CO2 Outlet Concentration (%)", value=2.0, min_value=0.0, max_value=100.0)
    column_height = st.number_input("Column Height (m)", value=10.0, min_value=0.0)
    cycle_time = st.number_input("Cycle Time (min)", value=30.0, min_value=0.0)
    pore_size = st.number_input("Pore Size (nm)", value=50.0, min_value=0.0)

# Additional parameters in expandable section
with st.expander("Advanced Parameters"):
    col3, col4 = st.columns(2)
    with col3:
        reactor_volume = st.number_input("Reactor Volume (m³)", value=5.0, min_value=0.0)
        reboiler_duty = st.number_input("Reboiler Duty (kW)", value=100.0, min_value=0.0)
        regeneration_temp = st.number_input("Regeneration Temp (°C)", value=120.0, min_value=0.0)
    with col4:
        residence_time = st.number_input("Residence Time (s)", value=60.0, min_value=0.0)
        stability = st.number_input("Stability (1-10)", value=8.0, min_value=1.0, max_value=10.0)
        surface_area = st.number_input("Surface Area (m²/g)", value=200.0, min_value=0.0)
        ph_level = st.number_input("pH Level", value=7.0, min_value=0.0, max_value=14.0)

# Create input data dictionary
input_data = {
    'Temperature (°C)': temperature,
    'Pressure (bar)': pressure,
    'CO2 Concentration (%)': co2_concentration,
    'Flow Rate (m³/h)': flow_rate,
    'Solvent Concentration (%)': solvent_concentration,
    'Sorbent Type': sorbent_type,
    'Capture Efficiency (%)': capture_efficiency,
    'Energy Consumption (MJ/kg CO2)': energy_consumption,
    'CO2 Outlet Concentration (%)': co2_outlet,
    'Column Height (m)': column_height,
    'Cycle Time (min)': cycle_time,
    'Pore Size (nm)': pore_size,
    'Reactor Volume (m³)': reactor_volume,
    'Reboiler Duty (kW)': reboiler_duty,
    'Regeneration Temp (°C)': regeneration_temp,
    'Residence Time (s)': residence_time,
    'Stability (1-10)': stability,
    'Surface Area (m²/g)': surface_area,
    'pH Level': ph_level
}

# Make prediction when button is clicked
if st.button("Predict", type="primary"):
    try:
        # Convert input to DataFrame
        df = pd.DataFrame([input_data])
        
        # Reorder columns to match model's feature names
        df = df[model_feature_names]
        
        # Encode categorical variables
        df['Sorbent Type'] = label_encoder.transform(df['Sorbent Type'])
        
        # Get numerical columns
        numerical_cols = [col for col in df.columns if col != 'Sorbent Type']
        
        # Preprocess numerical features
        df[numerical_cols] = imputer.transform(df[numerical_cols])
        df[numerical_cols] = scaler.transform(df[numerical_cols])
        
        # Make prediction
        predictions = model.predict(df)
        
        # Display predictions at the top
        st.markdown("### Predictions")
        col_pred1, col_pred2 = st.columns(2)
        
        with col_pred1:
            st.markdown("""
                <div class="metric-box">
                    <h4 class="metric-title">CO2 Emissions</h4>
                    <h2 style="color: #ff4b4b;">{:.2f} kg/hr</h2>
                </div>
            """.format(predictions[0][0]), unsafe_allow_html=True)
        
        with col_pred2:
            st.markdown("""
                <div class="metric-box">
                    <h4 class="metric-title">Energy Efficiency</h4>
                    <h2 style="color: #00acb5;">{:.2f} MJ</h2>
                </div>
            """.format(predictions[0][1]), unsafe_allow_html=True)
        
        # Add visualizations
        st.markdown("### Visualizations")
        viz_col1, viz_col2 = st.columns(2)
        
        with viz_col1:
            # Pie chart for CO2 distribution
            fig_pie = px.pie(
                values=[predictions[0][0], 100 - predictions[0][0]],
                names=['CO2 Emitted', 'CO2 Captured'],
                title='CO2 Distribution',
                color_discrete_sequence=['#ff4b4b', '#00acb5']
            )
            st.plotly_chart(fig_pie, use_container_width=True)
            
            # Bar chart for key parameters
            fig_bar = go.Figure(data=[
                go.Bar(
                    x=['Temperature', 'Pressure', 'CO2 Conc.', 'Flow Rate'],
                    y=[temperature, pressure, co2_concentration, flow_rate],
                    marker_color=['#ff4b4b', '#00acb5', '#ff4b4b', '#00acb5']
                )
            ])
            fig_bar.update_layout(title='Key Parameters')
            st.plotly_chart(fig_bar, use_container_width=True)
        
        with viz_col2:
            # Line chart for energy efficiency
            fig_line = go.Figure()
            fig_line.add_trace(go.Scatter(
                x=['Current', 'Target'],
                y=[predictions[0][1], 100],
                mode='lines+markers',
                name='Energy Efficiency'
            ))
            fig_line.update_layout(title='Energy Efficiency Trend')
            st.plotly_chart(fig_line, use_container_width=True)
            
            # Gauge chart for capture efficiency
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=capture_efficiency,
                title={'text': "Capture Efficiency (%)"},
                gauge={'axis': {'range': [0, 100]},
                       'bar': {'color': "#00acb5"}}
            ))
            st.plotly_chart(fig_gauge, use_container_width=True)
        
        # Add interpretation
        st.markdown("### Interpretation")
        if predictions[0][0] > 100:
            st.warning("High CO2 emissions detected. Consider reducing CO2 concentration or increasing capture efficiency.")
        if predictions[0][1] < 50:
            st.warning("Low energy efficiency detected. Consider optimizing process parameters.")
        
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        # Add debug information
        st.error("Debug Information:")
        st.write("Input DataFrame columns:", df.columns.tolist())
        st.write("Model feature names:", model_feature_names)

# Add footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center'>
        <p>Built by Vinay & Sai Teja</p>
    </div>
""", unsafe_allow_html=True) 