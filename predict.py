import pandas as pd
import numpy as np
import joblib
import warnings
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer

warnings.filterwarnings('ignore')

# List of features in the correct order (as in training)
FEATURE_ORDER = [
    'Temperature (°C)',
    'Pressure (bar)',
    'CO2 Concentration (%)',
    'Flow Rate (m³/h)',
    'Sorbent Type',
    'Surface Area (m²/g)',
    'Pore Size (nm)',
    'Stability (1-10)',
    'pH Level',
    'CO2 Outlet Concentration (%)',
    'Energy Consumption (MJ/kg CO2)',
    'Cycle Time (min)',
    'Capture Efficiency (%)',
    'Regeneration Temp (°C)',
    'Solvent Concentration (%)',
    'Reactor Volume (m³)',
    'Residence Time (s)',
    'Reboiler Duty (kW)',
    'Column Height (m)'
]

# List of numerical columns in the correct order (as in training, excluding 'Sorbent Type')
NUMERICAL_COLS = [
    'Temperature (°C)',
    'Pressure (bar)',
    'CO2 Concentration (%)',
    'Flow Rate (m³/h)',
    'Surface Area (m²/g)',
    'Pore Size (nm)',
    'Stability (1-10)',
    'pH Level',
    'CO2 Outlet Concentration (%)',
    'Energy Consumption (MJ/kg CO2)',
    'Cycle Time (min)',
    'Capture Efficiency (%)',
    'Regeneration Temp (°C)',
    'Solvent Concentration (%)',
    'Reactor Volume (m³)',
    'Residence Time (s)',
    'Reboiler Duty (kW)',
    'Column Height (m)'
]

def load_model_and_preprocessors():
    """Load the trained model and preprocessing objects."""
    try:
        model = joblib.load('carbon_emission_model.joblib')
        le = joblib.load('label_encoder.joblib')
        imputer = joblib.load('imputer.joblib')
        scaler = joblib.load('scaler.joblib')
        print("Model and preprocessing objects loaded successfully!")
        return model, le, imputer, scaler
    except FileNotFoundError as e:
        print(f"Error: Could not find model files. Please make sure you have run the training part of the code first.")
        print(f"Missing file: {str(e)}")
        return None, None, None, None

def preprocess_input_data(input_data, le, imputer, scaler):
    """Preprocess the input data using the saved preprocessing objects."""
    try:
        # Create DataFrame
        df_input = pd.DataFrame([input_data])
        print("\n[DEBUG] DataFrame before encoding and scaling:")
        print(df_input)
        print("[DEBUG] DataFrame columns:", df_input.columns.tolist())
        # Reorder columns to match training
        df_input = df_input[FEATURE_ORDER]
        print("[DEBUG] DataFrame after reordering columns:")
        print(df_input)
        # Encode categorical variables
        df_input['Sorbent Type'] = le.transform(df_input['Sorbent Type'])
        # Scale numerical features
        df_input[NUMERICAL_COLS] = imputer.transform(df_input[NUMERICAL_COLS])
        df_input[NUMERICAL_COLS] = scaler.transform(df_input[NUMERICAL_COLS])
        print("[DEBUG] DataFrame after encoding and scaling:")
        print(df_input)
        return df_input
    except Exception as e:
        import traceback
        print(f"Error during preprocessing: {str(e)}")
        traceback.print_exc()
        return None

def make_prediction(input_data):
    """Make predictions using the trained model."""
    # Load model and preprocessors
    model, le, imputer, scaler = load_model_and_preprocessors()
    if model is None:
        return None
    print("[DEBUG] Numerical columns:", NUMERICAL_COLS)
    # Preprocess input data
    processed_data = preprocess_input_data(input_data, le, imputer, scaler)
    if processed_data is None:
        return None
    # Make prediction
    try:
        print("[DEBUG] DataFrame columns before prediction:", processed_data.columns.tolist())
        prediction = model.predict(processed_data)
        return {
            'CO2 Emitted (kg/hr)': prediction[0][0],
            'Energy Used Efficiently (MJ)': prediction[0][1]
        }
    except Exception as e:
        import traceback
        print(f"Error during prediction: {str(e)}")
        traceback.print_exc()
        return None

def analyze_parameter_sensitivity(input_data, model, le, imputer, scaler, variation_percentage=10):
    """
    Analyze how each parameter affects CO2 emissions and energy efficiency.
    Returns recommendations for parameter adjustments.
    """
    base_prediction = make_prediction(input_data)
    if base_prediction is None:
        return None
    
    base_co2 = base_prediction['CO2 Emitted (kg/hr)']
    base_energy = base_prediction['Energy Used Efficiently (MJ)']
    
    recommendations = {
        'increase': [],
        'decrease': [],
        'neutral': []
    }
    
    for param in NUMERICAL_COLS:
        # Create variations of input data
        increased_data = input_data.copy()
        decreased_data = input_data.copy()
        
        # Calculate variations
        current_value = input_data[param]
        variation = current_value * (variation_percentage / 100)
        
        # Apply variations
        increased_data[param] = current_value + variation
        decreased_data[param] = current_value - variation
        
        # Get predictions for variations
        increased_pred = make_prediction(increased_data)
        decreased_pred = make_prediction(decreased_data)
        
        if increased_pred is None or decreased_pred is None:
            continue
        
        # Calculate impacts
        co2_increase_impact = increased_pred['CO2 Emitted (kg/hr)'] - base_co2
        co2_decrease_impact = decreased_pred['CO2 Emitted (kg/hr)'] - base_co2
        energy_increase_impact = increased_pred['Energy Used Efficiently (MJ)'] - base_energy
        energy_decrease_impact = decreased_pred['Energy Used Efficiently (MJ)'] - base_energy
        
        # Determine recommendation
        if co2_increase_impact < 0 and energy_increase_impact > 0:
            recommendations['increase'].append({
                'parameter': param,
                'co2_impact': abs(co2_increase_impact),
                'energy_impact': energy_increase_impact
            })
        elif co2_decrease_impact < 0 and energy_decrease_impact > 0:
            recommendations['decrease'].append({
                'parameter': param,
                'co2_impact': abs(co2_decrease_impact),
                'energy_impact': energy_decrease_impact
            })
        else:
            recommendations['neutral'].append({
                'parameter': param,
                'co2_impact': min(abs(co2_increase_impact), abs(co2_decrease_impact)),
                'energy_impact': max(energy_increase_impact, energy_decrease_impact)
            })
    
    return recommendations

def print_optimization_recommendations(recommendations):
    """Print formatted optimization recommendations in a concise way."""
    print("\nOptimization Recommendations for Better Performance:")
    print("-" * 50)
    
    if recommendations['increase']:
        print("\nINCREASE these parameters:")
        for rec in sorted(recommendations['increase'], key=lambda x: x['energy_impact'], reverse=True):
            print(f"• {rec['parameter']} (Current: {sample_input[rec['parameter']]})")
            print(f"  Expected improvement: {rec['co2_impact']:.2f} kg/hr less CO2, {rec['energy_impact']:.2f} MJ more efficiency")
    
    if recommendations['decrease']:
        print("\nDECREASE these parameters:")
        for rec in sorted(recommendations['decrease'], key=lambda x: x['energy_impact'], reverse=True):
            print(f"• {rec['parameter']} (Current: {sample_input[rec['parameter']]})")
            print(f"  Expected improvement: {rec['co2_impact']:.2f} kg/hr less CO2, {rec['energy_impact']:.2f} MJ more efficiency")
    
    # Only show neutral parameters if there are no increase/decrease recommendations
    if not recommendations['increase'] and not recommendations['decrease']:
        print("\nNo significant improvements found. Current parameters are well optimized.")

if __name__ == "__main__":
    # Example usage with all required features
    sample_input = {
        'Temperature (°C)': 85,
        'Pressure (bar)': 5,
        'Flow Rate (m³/h)': 1000,
        'CO2 Concentration (%)': 12,
        'CO2 Outlet Concentration (%)': 2,
        'Capture Efficiency (%)': 85,
        'Column Height (m)': 10,
        'Cycle Time (min)': 30,
        'Energy Consumption (MJ/kg CO2)': 2.5,
        'Pore Size (nm)': 50,
        'Reactor Volume (m³)': 5,
        'Reboiler Duty (kW)': 100,
        'Regeneration Temp (°C)': 120,
        'Residence Time (s)': 60,
        'Solvent Concentration (%)': 30,
        'Stability (1-10)': 8,
        'Surface Area (m²/g)': 200,
        'pH Level': 7,
        'Sorbent Type': 'MEA'
    }
    
    # Load model and preprocessors
    model, le, imputer, scaler = load_model_and_preprocessors()
    if model is not None:
        # Get base prediction
        result = make_prediction(sample_input)
        if result:
            print("\nCurrent Prediction Results:")
            print(f"Predicted CO2 Emitted: {result['CO2 Emitted (kg/hr)']:.2f} kg/hr")
            print(f"Predicted Energy Used Efficiently: {result['Energy Used Efficiently (MJ)']:.2f} MJ")
        
        # Get and print optimization recommendations
        recommendations = analyze_parameter_sensitivity(sample_input, model, le, imputer, scaler)
        if recommendations:
            print_optimization_recommendations(recommendations) 