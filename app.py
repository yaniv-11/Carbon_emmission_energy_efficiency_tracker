from flask import Flask, request, jsonify
from energy_efficiency_model import EnergyEfficiencyModel
import os

app = Flask(__name__)
model = EnergyEfficiencyModel()

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from request
        input_data = request.json
        
        # Make prediction
        result = model.predict(input_data)
        
        # Get optimization recommendations
        recommendations = model.analyze_parameter_sensitivity(input_data)
        
        # Format recommendations
        formatted_recommendations = {
            'increase': [
                {
                    'parameter': rec['parameter'],
                    'current_value': input_data[rec['parameter']],
                    'co2_improvement': rec['co2_impact'],
                    'energy_improvement': rec['energy_impact']
                }
                for rec in recommendations['increase']
            ],
            'decrease': [
                {
                    'parameter': rec['parameter'],
                    'current_value': input_data[rec['parameter']],
                    'co2_improvement': rec['co2_impact'],
                    'energy_improvement': rec['energy_impact']
                }
                for rec in recommendations['decrease']
            ]
        }
        
        # Prepare response
        response = {
            'predictions': result,
            'recommendations': formatted_recommendations
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    # Run the Flask app
    app.run(host='0.0.0.0', port=5000) 