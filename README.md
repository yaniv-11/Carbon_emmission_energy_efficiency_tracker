# Energy Efficiency Model

This model predicts CO2 emissions and energy efficiency for carbon capture processes based on various input parameters.

## Model Description

- **Model Type**: Multi-output XGBoost Regressor
- **Task**: Regression
- **Outputs**: 
  - CO2 Emissions (kg/hr)
  - Energy Efficiency (MJ)

## Input Parameters

- Temperature (°C)
- Pressure (bar)
- CO2 Concentration (%)
- Flow Rate (m³/h)
- Solvent Concentration (%)
- Sorbent Type (Activated Carbon, Zeolite, MOF, Amine)
- Capture Efficiency (%)
- Energy Consumption (MJ/kg CO2)
- CO2 Outlet Concentration (%)
- Column Height (m)
- Cycle Time (min)
- Pore Size (nm)
- Reactor Volume (m³)
- Reboiler Duty (kW)
- Regeneration Temp (°C)
- Residence Time (s)
- Stability (1-10)
- Surface Area (m²/g)
- pH Level

## Model Performance

- Overall Model Accuracy: ~85%
- Cross-validation R² scores:
  - CO2 Emissions: 0.85 (±0.03)
  - Energy Efficiency: 0.84 (±0.04)

## Usage

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import joblib
import pandas as pd

# Load model and preprocessors
model = joblib.load('carbon_emission_model.joblib')
label_encoder = joblib.load('label_encoder.joblib')
imputer = joblib.load('imputer.joblib')
scaler = joblib.load('scaler.joblib')

# Prepare input data
input_data = {
    'Temperature (°C)': 85,
    'Pressure (bar)': 5,
    'CO2 Concentration (%)': 12,
    # ... add other parameters
}

# Make prediction
df = pd.DataFrame([input_data])
df['Sorbent Type'] = label_encoder.transform(df['Sorbent Type'])
df[numerical_cols] = imputer.transform(df[numerical_cols])
df[numerical_cols] = scaler.transform(df[numerical_cols])
predictions = model.predict(df)

print(f"CO2 Emissions: {predictions[0][0]:.2f} kg/hr")
print(f"Energy Efficiency: {predictions[0][1]:.2f} MJ")
```

## Training Data

The model was trained on a dataset of 2000 samples with various carbon capture process parameters.

## Limitations

- The model assumes ideal operating conditions
- Predictions may not be accurate for extreme parameter values
- Performance may vary for new sorbent types not seen during training

## License

MIT License 
